class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        from itertools import combinations
        
        max_val = 0
        
        # Generate all possible subsequences of size 2 * k
        for subseq in combinations(nums, 2 * k):
            # Split the subsequence into two parts
            first_half = subseq[:k]
            second_half = subseq[k:]
            
            # Calculate the OR for both halves
            or_first_half = 0
            for num in first_half:
                or_first_half |= num
            
            or_second_half = 0
            for num in second_half:
                or_second_half |= num
            
            # Calculate the XOR of the OR results
            current_val = or_first_half ^ or_second_half
            
            # Update the maximum value found
            max_val = max(max_val, current_val)
        
        return max_val