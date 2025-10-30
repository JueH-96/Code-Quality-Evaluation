from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + possible[i]
        total_sum = prefix[-1]
        for k in range(1, n):
            alice_s = prefix[k]
            alice_score = 2 * alice_s - k
            bob_s = total_sum - alice_s
            bob_score = 2 * bob_s - (n - k)
            if alice_score > bob_score:
                return k
        return -1