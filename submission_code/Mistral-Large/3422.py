class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        a = [1] * n

        for _ in range(k):
            prefix_sum = 0
            new_a = [0] * n
            for i in range(n):
                prefix_sum += a[i]
                new_a[i] = prefix_sum
            a = new_a

        return a[-1] % MOD