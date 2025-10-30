from collections import Counter

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 10**9 + 7
        n = len(num)
        counts = Counter(num)
        memo = {}

        def solve(index, current_counts, diff):
            if index == n:
                return 1 if diff == 0 else 0
            
            state = (index, tuple(sorted(current_counts.items())), diff)
            if state in memo:
                return memo[state]
            
            ans = 0
            for digit_str in list(current_counts.keys()):
                digit = int(digit_str)
                if current_counts[digit_str] > 0:
                    next_counts = current_counts.copy()
                    next_counts[digit_str] -= 1
                    if next_counts[digit_str] == 0:
                        del next_counts[digit_str]
                    
                    if index % 2 == 0:
                        ans = (ans + solve(index + 1, next_counts, diff + digit)) % MOD
                    else:
                        ans = (ans + solve(index + 1, next_counts, diff - digit)) % MOD
            
            memo[state] = ans
            return ans

        return solve(0, counts, 0)