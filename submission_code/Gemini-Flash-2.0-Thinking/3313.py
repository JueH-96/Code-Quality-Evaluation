class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        memo = {}

        def solve(index, count):
            if (index, count) in memo:
                return memo[(index, count)]

            if count == 0:
                return 0

            if index >= n:
                return -float('inf')

            # Option 1: Don't start a subarray at index
            res = solve(index + 1, count)

            # Option 2: Start a subarray at index
            for end in range(index, n):
                subarray_sum = sum(nums[index : end + 1])

                # Consider this subarray as the 'count'-th subarray in the strength calculation
                coeff_index = k - count + 1
                coeff = coeff_index if count % 2 == 1 else -coeff_index

                remaining_strength = solve(end + 1, count - 1)

                if remaining_strength != -float('inf'):
                    res = max(res, subarray_sum * coeff + remaining_strength)

            memo[(index, count)] = res
            return res

        return solve(0, k)