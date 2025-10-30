from typing import List

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        # Create a list of (value, original_index) pairs.
        pairs = [(val, i) for i, val in enumerate(nums)]
        pairs.sort(key=lambda x: x[0])
        
        # The idea is that two numbers can be "chain‐swappable" if there is a sequence
        # of numbers between them (in terms of value) whose consecutive differences are <= limit.
        # When we sort the numbers by their value, the connected components (by allowed swaps)
        # appear as contiguous segments. Within each such segment we can arbitrarily reassign the numbers
        # among the original indices that appear in that component.
        
        res = [0] * n  # This will hold the final answer.
        i = 0
        while i < n:
            # Start a new connected component (group) from the sorted list.
            group = []
            group.append(pairs[i])
            j = i + 1
            # Collect as long as the difference between consecutive sorted values is <= limit.
            while j < n and pairs[j][0] - pairs[j - 1][0] <= limit:
                group.append(pairs[j])
                j += 1
            
            # For the current group, extract the original indices and their values.
            indices = [idx for (_, idx) in group]
            values = [val for (val, _) in group]
            
            # To achieve the lexicographically smallest overall array, 
            # assign the smallest available value in the group to the smallest index (position)
            # that belongs to that group.
            indices.sort()  # sort indices in ascending order (positions in the array)
            values.sort()   # sort values in ascending order
            
            for k in range(len(indices)):
                res[indices[k]] = values[k]
            
            # Move to the next group.
            i = j
        
        return res