from typing import List

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        # Step 1: Find the dominant element using Boyer-Moore Voting.
        candidate = None
        count = 0
        for val in nums:
            if count == 0:
                candidate = val
                count = 1
            elif candidate == val:
                count += 1
            else:
                count -= 1
        
        # candidate is our potential dominant element
        # Step 2: Count how many times candidate appears.
        freq_candidate = sum(1 for x in nums if x == candidate)
        
        n = len(nums)
        
        # Step 3: Build prefix counts for the candidate.
        prefix_count = [0] * n
        prefix_count[0] = 1 if nums[0] == candidate else 0
        for i in range(1, n):
            prefix_count[i] = prefix_count[i - 1] + (1 if nums[i] == candidate else 0)
        
        # Step 4: Check each valid split index.
        for i in range(n - 1):
            left_count = prefix_count[i]
            left_len = i + 1
            right_count = freq_candidate - left_count
            right_len = n - left_len
            
            # The candidate is dominant in each subarray if it appears more
            # than half of that subarray's length.
            if left_count * 2 > left_len and right_count * 2 > right_len:
                return i
        
        return -1