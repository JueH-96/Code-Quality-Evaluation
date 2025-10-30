class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + possible[i]
        total = prefix[-1]
        for i in range(n-1):
            sum_alice = prefix[i+1]
            alice_score = 2 * sum_alice - (i + 1)
            sum_bob = total - sum_alice
            bob_score = 2 * sum_bob - (n - i - 1)
            if alice_score > bob_score:
                return i + 1
        return -1