class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        
        # Compute total score: +1 for clearable level, -1 for impossible level
        total_score = sum(2 * p - 1 for p in possible)
        
        alice_score = 0
        for k in range(1, n):  # Alice plays k levels (0 to k-1), Bob plays n-k levels (k to n-1)
            alice_score += 2 * possible[k-1] - 1
            if 2 * alice_score > total_score:  # alice_score > (total_score - alice_score)
                return k
        
        return -1