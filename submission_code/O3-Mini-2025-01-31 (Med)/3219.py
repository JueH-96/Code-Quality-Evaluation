from typing import List

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        # First, pair each element with its original index.
        pairs = [(num, i) for i, num in enumerate(nums)]
        # Sort by the number; this helps us group numbers that can be connected.
        pairs.sort(key=lambda x: x[0])
        
        # The key observation is:
        # In one allowed operation, you can swap two indices i and j when |nums[i] - nums[j]| <= limit.
        # This defines a graph among the indices where an edge exists between two indices if their 
        # values differ by at most 'limit'. By transitivity, within a connected component you can
        # arbitrarily rearrange the numbers.
        # Moreover, if we sort the numbers by value, then any connected component will appear as a 
        # contiguous block. In particular, if in sorted order the difference between two adjacent
        # numbers is greater than 'limit', then they cannot be connected (even by an intermediate chain).
        
        # Group the sorted pairs into connected components.
        components = []
        current_component = [pairs[0]]  # start with first pair
        for i in range(1, n):
            # if the gap between current and previous number is <= limit, they are connected
            if pairs[i][0] - pairs[i-1][0] <= limit:
                current_component.append(pairs[i])
            else:
                components.append(current_component)
                current_component = [pairs[i]]
        components.append(current_component)
        
        # Create a result array. For each connected component, we can reorder arbitrarily.
        # To get lexicographically smallest overall array, for each component we assign its smallest 
        # available number to the smallest index in that component.
        res = list(nums)
        for comp in components:
            # Get the list of original indices and the corresponding values from the component.
            indices = [idx for (_, idx) in comp]
            values = [val for (val, _) in comp]
            # Sort indices so that the smallest index gets the smallest value.
            indices.sort()
            # Sort the values.
            values.sort()
            # Place the sorted values back to the corresponding indices.
            for pos, idx in enumerate(indices):
                res[idx] = values[pos]
                
        return res

# When running the code, you can test it as follows:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.lexicographicallySmallestArray([1, 5, 3, 9, 8], 2))
    # Expected output: [1, 3, 5, 8, 9]
    
    # Example 2:
    print(sol.lexicographicallySmallestArray([1, 7, 6, 18, 2, 1], 3))
    # Expected output: [1, 6, 7, 18, 1, 2]
    
    # Example 3:
    print(sol.lexicographicallySmallestArray([1, 7, 28, 19, 10], 3))
    # Expected output: [1, 7, 28, 19, 10]