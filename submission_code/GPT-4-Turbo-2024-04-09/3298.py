class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count the frequency of each number and each number + 1
        freq = Counter(nums)
        freq_plus_one = Counter(n + 1 for n in nums)
        
        # To store the maximum length of consecutive elements
        max_consecutive = 0
        
        # Iterate through each unique number in the original list
        for num in set(nums):
            # If num is the start of a consecutive sequence
            if num - 1 not in freq:
                # Count the length of the consecutive sequence starting from num
                length = 0
                while num in freq or num in freq_plus_one:
                    length += 1
                    num += 1
                # Update the maximum length found
                max_consecutive = max(max_consecutive, length)
        
        return max_consecutive