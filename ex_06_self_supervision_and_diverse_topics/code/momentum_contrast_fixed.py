import numpy as np
import collections
import torch
import torch.nn as nn

class LimitedSamplingQueue():
    def __init__(self, max_size):
        self.queue = collections.deque(maxlen=max_size)
        self.max_size = max_size
        
    def add_items(self, items):
        # CRITICAL FIX: Add individual feature vectors, not the whole batch
        for item in items:
            self.queue.append(item)
    
    def remove_items(self, num_items):
        for i in range(min(num_items, len(self.queue))):
            self.queue.popleft()
    
    def sample_items(self, num_items):
        if len(self.queue) == 0:
            return []
        
        actual_num = min(num_items, len(self.queue))
        sample_indices = np.random.randint(low=0, high=len(self.queue), size=actual_num)
        return [self.queue[idx] for idx in sample_indices]
    
    def len(self):
        return len(self.queue)


def momentum_contrast_loss(q, k, negative_samples_queue, num_neg_sampl=10, temperature=0.07):
    B, M = q.shape
    
    # L2 normalize features - THIS IS CRITICAL for contrastive learning!
    q = torch.nn.functional.normalize(q, dim=1)
    k = torch.nn.functional.normalize(k, dim=1)
    
    # Compute positive logits: (B, 1, M) @ (B, M, 1) -> (B, 1, 1) -> (B, 1)
    pos_logits = (q.unsqueeze(1) @ k.unsqueeze(2)).squeeze(2)  # (B, 1)
    
    # Sample negative samples from queue
    neg_samples_list = negative_samples_queue.sample_items(num_neg_sampl)
    if len(neg_samples_list) == 0:
        # If queue is empty, use other samples in the batch as negatives
        neg_samples = k
    else:
        neg_samples = torch.stack(neg_samples_list).cuda()  # (num_neg_sampl, M)
        # Normalize negative samples
        neg_samples = torch.nn.functional.normalize(neg_samples, dim=1)
    
    # Compute negative logits: (B, M) @ (M, num_neg_sampl) -> (B, num_neg_sampl)
    negative_logits = q @ neg_samples.T  # FIXED: Use transpose!
    
    # Concatenate logits
    logits = torch.cat([pos_logits, negative_logits], dim=1)  # (B, 1 + num_neg_sampl)
    
    # Apply temperature scaling
    logits = logits / temperature
    
    # Labels: positive samples are at index 0
    labels = torch.zeros(B, dtype=torch.long).cuda()
    
    # Compute cross-entropy loss
    loss = torch.nn.functional.cross_entropy(logits, labels)
    
    return loss


class MomentumContrast(nn.Module):
    def __init__(self, key_encoder, query_encoder, projector, key_queue_max_size=256, num_neg_samples=100):
        super().__init__()
        self.key_encoder = key_encoder
        self.query_encoder = query_encoder
        self.projector = projector
        self.negative_samples_queue = LimitedSamplingQueue(max_size=key_queue_max_size)
        self.num_neg_samples = num_neg_samples
        
        # Initialize key encoder with query encoder weights
        for param_q, param_k in zip(query_encoder.parameters(), key_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
    
    def forward(self, y1, y2, warmup=False):
        # Query features from y2
        q = self.projector(self.query_encoder(y2))
        
        # Key features from y1 (no gradients)
        with torch.no_grad():
            k = self.projector(self.key_encoder(y1))
        
        if warmup:
            # CRITICAL FIX: Add individual feature vectors to queue
            k_list = [k[i].detach().cpu() for i in range(k.size(0))]
            self.negative_samples_queue.add_items(k_list)
            return torch.tensor(0.0).cuda()
        else:
            # Compute loss
            loss = momentum_contrast_loss(q, k, self.negative_samples_queue, self.num_neg_samples)
            
            # CRITICAL FIX: Add individual feature vectors to queue
            with torch.no_grad():
                k_list = [k[i].detach().cpu() for i in range(k.size(0))]
                self.negative_samples_queue.add_items(k_list)
            
            return loss
