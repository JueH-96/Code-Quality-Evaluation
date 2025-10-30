class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        req = {}
        for end, cnt in requirements:
            req[end] = cnt

        dp = {}

        def count_inversions(perm):
            count = 0
            for i in range(len(perm)):
                for j in range(i + 1, len(perm)):
                    if perm[i] > perm[j]:
                        count += 1
            return count

        def solve(index, current_perm):
            if index == n:
                for end, cnt in req.items():
                    if count_inversions(current_perm[:end + 1]) != cnt:
                        return 0
                return 1

            state = (index, tuple(sorted(current_perm)))
            if state in dp:
                return dp[state]

            ans = 0
            for num in range(n):
                if num not in current_perm:
                    ans = (ans + solve(index + 1, current_perm + [num])) % MOD

            dp[state] = ans
            return ans

        return solve(0, [])