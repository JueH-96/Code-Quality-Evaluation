class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        
        # Calculate the total score of all levels
        total_score = sum(1 if p == 1 else -1 for p in possible)
        
        alice_score = 0
        
        for i in range(n-1):  # Try all splitting points, ensuring Bob plays at least 1 level
            # Add the current level's score to Alice's total
            alice_score += 1 if possible[i] == 1 else -1
            
            # Bob's score is what remains
            bob_score = total_score - alice_score
            
            # Check if Alice has more points than Bob
            if alice_score > bob_score:
                return i + 1  # Return the number of levels Alice played
        
        # If no valid split was found
        return -1