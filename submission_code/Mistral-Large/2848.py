from typing import List

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        memo = {}

        def is_special(perm):
            for i in range(len(perm) - 1):
                if perm[i] % perm[i + 1] != 0 and perm[i + 1] % perm[i] != 0:
                    return False
            return True

        def backtrack(used, path):
            if len(path) == n:
                if is_special(path):
                    return 1
                return 0

            key = tuple(used) + tuple(path)
            if key in memo:
                return memo[key]

            count = 0
            for i in range(n):
                if not used[i]:
                    used[i] = True
                    path.append(nums[i])
                    count += backtrack(used, path)
                    count %= MOD
                    path.pop()
                    used[i] = False

            memo[key] = count
            return count

        used = [False] * n
        return backtrack(used, [])

# Example usage:
# sol = Solution()
# print(sol.specialPerm([2, 3, 6]))  # Output: 2
# print(sol.specialPerm([1, 4, 3]))  # Output: 2