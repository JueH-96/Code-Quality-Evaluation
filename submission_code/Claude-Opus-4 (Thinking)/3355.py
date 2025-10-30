class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        
        # Convert to scores: 1 → 1, 0 → -1
        scores = [1 if p == 1 else -1 for p in possible]
        
        # Calculate total score
        total_score = sum(scores)
        
        # Try each possible split
        alice_score = 0
        for k in range(1, n):  # Alice plays levels 0 to k-1, Bob plays k to n-1
            alice_score += scores[k - 1]
            bob_score = total_score - alice_score
            
            if alice_score > bob_score:
                return k
        
        return -1