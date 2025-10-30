from typing import List
# Using sortedcontainers library which provides SortedList.
# This library is usually available in competitive programming environments.
from sortedcontainers import SortedList

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total_imbalance = 0

        # Iterate through all possible left endpoints 'i' of subarrays
        for i in range(n):
            # For each starting point 'i', we build subarrays ending at 'j'
            sl = SortedList()  # Stores elements of the current subarray nums[i...j]
            current_imbalance = 0  # Imbalance count for the current subarray nums[i...j]
            
            # Iterate through all possible right endpoints 'j' of subarrays
            for j in range(i, n):
                val = nums[j]  # The new element being added to the subarray

                # Find the index where 'val' would be inserted in the sorted list 'sl'.
                # This 'idx' also tells us its potential neighbors *before* insertion.
                idx = sl.bisect_left(val)
                
                # Determine the element immediately before 'val' (prev_val)
                # and immediately after 'val' (next_val) in the sorted list *before* 'val' is added.
                # We use -1 to indicate non-existence for simpler checks.
                prev_val = sl[idx-1] if idx > 0 else -1
                next_val = sl[idx] if idx < len(sl) else -1
                
                # Now, calculate how the imbalance number changes by adding 'val':
                
                # Rule 1: An existing imbalance gap (prev_val, next_val) might be broken by 'val'.
                # This occurs if prev_val and next_val were present, and they formed a gap
                # (next_val - prev_val > 1). After 'val' is inserted between them, this single
                # gap is effectively replaced by two potentially smaller gaps (prev_val, val)
                # and (val, next_val). So, we decrement the count of the broken gap.
                if prev_val != -1 and next_val != -1:
                    if next_val - prev_val > 1:
                        current_imbalance -= 1
                
                # Rule 2: A new imbalance gap might be formed between prev_val and 'val'.
                # This occurs if prev_val exists and val is more than 1 greater than prev_val.
                if prev_val != -1:
                    if val - prev_val > 1:
                        current_imbalance += 1
                        
                # Rule 3: A new imbalance gap might be formed between 'val' and next_val.
                # This occurs if next_val exists and next_val is more than 1 greater than val.
                if next_val != -1:
                    if next_val - val > 1:
                        current_imbalance += 1
                
                # Add 'val' to the sorted list 'sl'.
                sl.add(val)
                
                # Add the imbalance of the current subarray (nums[i...j]) to the total sum.
                total_imbalance += current_imbalance
                
        return total_imbalance