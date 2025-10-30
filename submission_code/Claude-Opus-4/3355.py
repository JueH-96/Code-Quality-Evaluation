class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        
        # Convert to scores: 1 stays 1, 0 becomes -1
        scores = [1 if x == 1 else -1 for x in possible]
        
        # Calculate total score
        total_score = sum(scores)
        
        # Alice must play at least 1 level, Bob must play at least 1 level
        # So Alice can play from 1 to n-1 levels
        alice_score = 0
        
        for k in range(1, n):  # k is the number of levels Alice plays
            alice_score += scores[k-1]
            bob_score = total_score - alice_score
            
            if alice_score > bob_score:
                return k
        
        return -1