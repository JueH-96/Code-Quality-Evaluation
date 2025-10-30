class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        """
        We want to produce the lexicographically smallest array obtainable by swapping
        elements whose absolute difference is <= limit, any number of times.

        Key observation:
        If we can swap values x and y (|x - y| <= limit), and y and z (|y - z| <= limit),
        then we can effectively move x to the position of z through a sequence of swaps.
        In sorted order, this transitivity implies that a set of consecutive values whose
        pairwise adjacent differences are <= limit can all travel among each other's positions.

        Concretely:
         1) Pair up each element value with its original index: (value, index).
         2) Sort by the value.
         3) Walk through the sorted list. Whenever consecutive sorted values differ by <= limit,
            they belong to the same connected "block." Otherwise, start a new block.
         4) Each block thus forms a connected component (by transitivity of swapping through
            intermediate values).
         5) Within each block:
              - Collect all original indices of that block.
              - Sort those indices.
              - Sort the block's values.
              - Place the sorted values back into the array positions corresponding to those sorted indices.
        
        This yields the lexicographically smallest arrangement possible under the allowed swaps.
        """

        n = len(nums)
        # Pair up (value, original_index)
        arr = [(val, i) for i, val in enumerate(nums)]
        # Sort by value
        arr.sort(key=lambda x: x[0])  # O(n log n)

        # We'll build blocks of consecutive elements in this sorted list
        # where consecutive values differ by <= limit.
        result = [0]*n

        start = 0
        for i in range(1, n+1):
            # Either we've reached the end or we found a gap > limit
            if i == n or (arr[i][0] - arr[i-1][0] > limit):
                # Everything from arr[start] to arr[i-1] (inclusive) is in one "block"
                block = arr[start:i]
                # Extract indices and values of this block
                block_indices = [x[1] for x in block]
                block_values  = [x[0] for x in block]
                # Sort them
                block_indices.sort()
                block_values.sort()
                # Assign sorted values to those sorted positions
                for idx, val in zip(block_indices, block_values):
                    result[idx] = val
                start = i

        return result