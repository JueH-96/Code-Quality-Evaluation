class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # Create a dictionary to count the frequency of each possible value
        count = defaultdict(int)
        
        # Iterate over each number in nums
        for num in nums:
            # Increment the count for each possible value in the range [num - k, num + k]
            for value in range(num - k, num + k + 1):
                count[value] += 1
        
        # The maximum beauty is the highest frequency found in the count dictionary
        return max(count.values())