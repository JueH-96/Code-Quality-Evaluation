from typing import List

class BIT:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

    def update(self, idx, delta):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        unique_sorted = sorted(set(nums))
        m = len(unique_sorted)
        compression = {v: i for i, v in enumerate(unique_sorted)}
        bit = BIT(m)
        present_ranks = set()
        min_diff = float('inf')
        n = len(nums)
        
        for j in range(n):
            if j >= x:
                val_to_add = nums[j - x]
                r = compression[val_to_add]
                if r not in present_ranks:
                    present_ranks.add(r)
                    bit.update(r + 1, 1)  # 1-based index
            
            current_val = nums[j]
            if len(present_ranks) == 0:
                continue
            
            target_rank = compression[current_val]
            
            # Find predecessor
            low_p, high_p = 0, m - 1
            best_p = -1
            while low_p <= high_p:
                mid_p = (low_p + high_p) // 2
                sum_mid_p = bit.query(mid_p + 1)
                if sum_mid_p > 0:
                    best_p = mid_p
                    low_p = mid_p + 1
                else:
                    high_p = mid_p - 1
            pred_val = None
            if best_p != -1:
                pred_val = unique_sorted[best_p]
            
            # Find successor
            low_s, high_s = target_rank, m - 1
            best_s = -1
            while low_s <= high_s:
                mid_s = (low_s + high_s) // 2
                sum_mid_s = bit.query(mid_s + 1)
                sum_before_s = bit.query(target_rank)
                count = sum_mid_s - sum_before_s
                if count > 0:
                    best_s = mid_s
                    high_s = mid_s - 1
                else:
                    low_s = mid_s + 1
            succ_val = None
            if best_s != -1:
                succ_val = unique_sorted[best_s]
            
            # Update min_diff
            if pred_val is not None:
                min_diff = min(min_diff, abs(current_val - pred_val))
            if succ_val is not None:
                min_diff = min(min_diff, abs(current_val - succ_val))
        
        return min_diff