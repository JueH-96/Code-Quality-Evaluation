from typing import List
from collections import defaultdict, Counter

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        # Step 1: Find the dominant element of the original array
        count = Counter(nums)
        d = None
        for num, freq in count.items():
            if freq * 2 > len(nums):
                d = num
                break
        
        n = len(nums)
        if n == 0:
            return -1  # According to constraints, n >=1
        
        # Step 2: Precompute prefix_dominant and max_other_left
        prefix_dominant = [0] * n
        max_other_left = [0] * n
        other_freq = defaultdict(int)
        current_max_other = 0
        
        for i in range(n):
            if nums[i] == d:
                prefix_dominant[i] = (prefix_dominant[i-1] if i > 0 else 0) + 1
            else:
                other_freq[nums[i]] += 1
                if other_freq[nums[i]] > current_max_other:
                    current_max_other = other_freq[nums[i]]
                prefix_dominant[i] = (prefix_dominant[i-1] if i > 0 else 0)
            max_other_left[i] = current_max_other
        
        # Step 3: Precompute suffix_dominant and max_other_right
        suffix_dominant = [0] * n
        max_other_right = [0] * n
        other_freq_suffix = defaultdict(int)
        current_max_other_suffix = 0
        
        for i in range(n-1, -1, -1):
            if nums[i] == d:
                suffix_dominant[i] = (suffix_dominant[i+1] if i < n-1 else 0) + 1
            else:
                other_freq_suffix[nums[i]] += 1
                if other_freq_suffix[nums[i]] > current_max_other_suffix:
                    current_max_other_suffix = other_freq_suffix[nums[i]]
                suffix_dominant[i] = (suffix_dominant[i+1] if i < n-1 else 0)
            max_other_right[i] = current_max_other_suffix
        
        # Step 4: Iterate through possible split indices
        min_index = -1
        for i in range(n-1):
            # Check left subarray (0..i)
            left_count = prefix_dominant[i]
            left_len = i + 1
            cond1 = (left_count * 2) > left_len
            cond2 = (max_other_left[i] * 2) <= left_len
            
            # Check right subarray (i+1..n-1)
            right_count = suffix_dominant[i+1]
            right_len = n - (i + 1)
            cond3 = (right_count * 2) > right_len
            cond4 = (max_other_right[i+1] * 2) <= right_len
            
            if cond1 and cond2 and cond3 and cond4:
                if min_index == -1 or i < min_index:
                    min_index = i
        
        return min_index