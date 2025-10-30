class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_or = [0] * (n + 1)

        # Calculate the prefix OR array
        for i in range(n):
            prefix_or[i + 1] = prefix_or[i] | nums[i]

        min_diff = float('inf')

        # Iterate through all possible subarrays
        for l in range(n):
            for r in range(l, n):
                subarray_or = prefix_or[r + 1] ^ prefix_or[l]
                min_diff = min(min_diff, abs(k - subarray_or))

        return min_diff