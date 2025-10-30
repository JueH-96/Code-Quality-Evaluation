from typing import List

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        """
        The idea is to look only at the numbers that are present – not at their
        current positions.  If two numbers differ by at most `limit` they can be
        swapped, thus they belong to the same *value*-component.  Because we can
        repeat the operation, every value inside one component can finally reach
        every index that initially contained a value from the same component.
        
        1.  Sort all (value, index) pairs by the value.
        2.  Scan that sorted list and split it into consecutive segments where
            the difference between neighbouring values is ≤ limit.  All pairs
            in one segment form one component.
        3.  Inside each component we can rearrange the values arbitrarily among
            the corresponding indices.  To obtain the lexicographically
            smallest arrangement we
               • sort the indices of the component,
               • keep the values in ascending order (they are already sorted),
               • place the smallest value at the smallest index, the next one
                 at the next index, and so on.
        4.  Concatenate the results of all components.
        
        Complexity:
            Sorting  : O(n log n)
            Processing components: O(n)
            Extra memory          : O(n)
        """
        n = len(nums)
        # (value, index) pairs sorted by value
        pairs = sorted(((val, idx) for idx, val in enumerate(nums)))
        
        res = [0] * n
        start = 0
        
        while start < n:
            end = start
            # enlarge the current component as long as the next value is close enough
            while end + 1 < n and pairs[end + 1][0] - pairs[end][0] <= limit:
                end += 1
            
            # collect indices and (already sorted) values of this component
            indices = [pairs[i][1] for i in range(start, end + 1)]
            values  = [pairs[i][0] for i in range(start, end + 1)]
            
            indices.sort()               # to minimize lexicographic order
            for idx, val in zip(indices, values):
                res[idx] = val
            
            start = end + 1
        
        return res