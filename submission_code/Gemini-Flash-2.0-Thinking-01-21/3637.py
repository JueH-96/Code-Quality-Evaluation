from collections import Counter

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        velunexorai = num
        digit_counts = Counter(velunexorai)
        n = len(velunexorai)
        total_sum = sum(int(d) for d in velunexorai)
        if total_sum % 2 != 0:
            return 0
        target_sum = total_sum // 2
        MOD = 10**9 + 7
        memo = {}

        def solve(index, diff, counts_tuple):
            if index == n:
                return 1 if diff == 0 else 0
            if (index, diff, counts_tuple) in memo:
                return memo[(index, diff, counts_tuple)]

            ans = 0
            counts_list = list(counts_tuple)
            for digit in range(10):
                if counts_list[digit] > 0:
                    counts_list[digit] -= 1
                    next_counts_tuple = tuple(counts_list)
                    if index % 2 == 0:
                        ans = (ans + solve(index + 1, diff + digit, next_counts_tuple)) % MOD
                    else:
                        ans = (ans + solve(index + 1, diff - digit, next_counts_tuple)) % MOD
                    counts_list[digit] += 1
            memo[(index, diff, counts_tuple)] = ans
            return ans

        initial_counts_tuple = tuple(digit_counts[str(d)] for d in range(10))
        result = solve(0, 0, initial_counts_tuple)
        return result