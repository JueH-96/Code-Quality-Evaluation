class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        # Create a dictionary to store the count of each element in the array
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        # Create a list of the distinct counts and sort it
        distinct_counts = sorted(set(count.values()))
        
        # Find the median of the distinct counts
        n = len(distinct_counts)
        if n % 2 == 0:
            return distinct_counts[n // 2 - 1]
        else:
            return distinct_counts[n // 2]