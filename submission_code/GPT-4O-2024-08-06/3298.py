class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count the frequency of each number in the array
        freq = Counter(nums)
        
        # Initialize the maximum length of consecutive elements
        max_consecutive = 0
        
        # Iterate over each unique number in the array
        for num in freq:
            # Calculate the possible maximum consecutive length
            # by considering the current number, the number + 1, and the number + 2
            current_consecutive = freq[num] + freq.get(num + 1, 0) + freq.get(num + 2, 0)
            # Update the maximum length found
            max_consecutive = max(max_consecutive, current_consecutive)
        
        return max_consecutive