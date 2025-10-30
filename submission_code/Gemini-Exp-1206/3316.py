class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        mod = 10**9 + 7

        def power(subsequence):
            min_diff = float('inf')
            for i in range(len(subsequence)):
                for j in range(i + 1, len(subsequence)):
                    min_diff = min(min_diff, abs(subsequence[i] - subsequence[j]))
            return min_diff if min_diff != float('inf') else 0

        def combinations(arr, k):
            result = []
            def backtrack(combination, start):
                if len(combination) == k:
                    result.append(combination[:])
                    return
                for i in range(start, len(arr)):
                    combination.append(arr[i])
                    backtrack(combination, i + 1)
                    combination.pop()
            backtrack([], 0)
            return result

        subsequences = combinations(nums, k)
        total_power = 0
        for subsequence in subsequences:
            total_power = (total_power + power(subsequence)) % mod
        return total_power