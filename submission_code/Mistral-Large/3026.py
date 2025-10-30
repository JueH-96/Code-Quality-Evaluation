class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        used = set()
        current_sum = 0
        num = 1

        while len(used) < n:
            if (target - num) not in used:
                used.add(num)
                current_sum = (current_sum + num) % MOD
            num += 1

        return current_sum