from collections import defaultdict
from sortedcontainers import SortedList

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        # Create a dictionary to hold SortedLists of numbers within the limit range
        buckets = defaultdict(SortedList)
        
        # Distribute numbers into appropriate buckets
        for num in nums:
            buckets[num // (limit + 1)].add(num)
        
        # Prepare the result array
        result = []
        
        # Iterate through the buckets and append the sorted numbers to the result
        for bucket in buckets.values():
            for num in bucket:
                result.append(num)
        
        return result