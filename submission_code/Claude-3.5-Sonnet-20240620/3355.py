class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        alice_score = 0
        bob_score = 0
        
        # Calculate Bob's score if he plays all levels
        for i in range(n):
            bob_score += 1 if possible[i] == 1 else -1
        
        # Iterate through levels and calculate scores
        for i in range(n):
            if possible[i] == 1:
                alice_score += 1
                bob_score -= 1
            else:
                alice_score -= 1
                bob_score += 1
            
            # Check if Alice's score is greater than Bob's
            if alice_score > bob_score:
                return i + 1
        
        # If Alice can't gain more points, return -1
        return -1