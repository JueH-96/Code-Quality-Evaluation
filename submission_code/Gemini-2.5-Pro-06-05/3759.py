from typing import List

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        
        MAX_VAL = 10**6

        class BIT:
            """Binary Indexed Tree for 1-based indexing."""
            def __init__(self, size: int):
                self.tree = [0] * (size + 1)
                self.size = size + 1
                self.max_pow = (size).bit_length() - 1

            def update(self, idx: int, delta: int):
                while idx < self.size:
                    self.tree[idx] += delta
                    idx += idx & (-idx)
            
            def query(self, idx: int) -> int:
                s = 0
                while idx > 0:
                    s += self.tree[idx]
                    idx -= idx & (-idx)
                return s

            def find_kth(self, k_val: int) -> int:
                """Finds the smallest 1-based index `v` such that query(v) >= k_val."""
                if k_val <= 0:
                    return 0
                
                idx = 0
                for p in range(self.max_pow, -1, -1):
                    step = 1 << p
                    if idx + step < self.size and self.tree[idx + step] < k_val:
                        k_val -= self.tree[idx + step]
                        idx += step
                
                return idx + 1

        bit_count = BIT(MAX_VAL)
        bit_sum = BIT(MAX_VAL)

        n = len(nums1)
        items = sorted([(nums1[i], nums2[i], i) for i in range(n)])
        
        answer = [0] * n
        total_count = 0
        total_sum = 0
        
        i = 0
        while i < n:
            # Identify a block of items with the same nums1 value
            j = i
            while j < n and items[j][0] == items[i][0]:
                j += 1
            
            # Calculate the top-k sum for all items in the current block.
            # The calculation is based on the state before this block's items are added.
            current_top_k_sum = 0
            if total_count > k:
                to_discard = total_count - k
                # Find the threshold value: the value of the to_discard-th smallest element
                v_thresh = bit_count.find_kth(to_discard)
                
                # Sum of elements strictly greater than v_thresh
                sum_gt_thresh = total_sum - bit_sum.query(v_thresh)
                
                # Number of elements less than or equal to v_thresh
                count_le_thresh = bit_count.query(v_thresh)
                
                # Number of elements with value v_thresh to keep
                to_keep_at_thresh = count_le_thresh - to_discard
                
                current_top_k_sum = sum_gt_thresh + to_keep_at_thresh * v_thresh
            else:  # total_count <= k, so we take all available values
                current_top_k_sum = total_sum
            
            # Assign this sum to all items in the current block
            for l in range(i, j):
                original_index = items[l][2]
                answer[original_index] = current_top_k_sum
            
            # Add nums2 values from the current block to the data structures
            for l in range(i, j):
                v2 = items[l][1]
                bit_count.update(v2, 1)
                bit_sum.update(v2, v2)
                total_count += 1
                total_sum += v2
                
            # Move to the next block
            i = j
            
        return answer