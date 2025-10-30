class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        # Pair each number with its index
        indexed_nums = [(nums[i], i) for i in range(n)]
        # Sort based on the number
        indexed_nums.sort()
        result = [0] * n
        i = 0
        while i < n:
            # Find the range where the difference between consecutive elements is <= limit
            j = i
            while j + 1 < n and indexed_nums[j+1][0] - indexed_nums[j][0] <= limit:
                j += 1
            # Extract the indices in this range
            indices = [indexed_nums[k][1] for k in range(i, j+1)]
            # Sort the indices to place the smallest number in the earliest possible position
            indices.sort()
            # Assign the sorted numbers to the sorted indices
            for k in range(i, j+1):
                result[indices[k-i]] = indexed_nums[k][0]
            i = j + 1
        return result