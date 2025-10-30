from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        # Initialize prefix count and frequency map
        prefix_cnt = 0
        freq = {0: 1}
        answer = 0
        
        # Iterate through each element in nums
        for num in nums:
            # Update prefix_cnt based on the condition
            if num % modulo == k:
                prefix_cnt += 1
            # Calculate the required prefix count for the condition
            required = (prefix_cnt % modulo - k + modulo) % modulo
            # Add the frequency of the required prefix count to the answer
            answer += freq.get(required, 0)
            # Update the frequency map with the current prefix count
            freq[prefix_cnt % modulo] = freq.get(prefix_cnt % modulo, 0) + 1
        
        return answer