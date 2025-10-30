class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        
        # Calculate total points if Alice plays all levels
        total_points = sum(possible)
        
        # If all levels are impossible, return -1
        if total_points == 0:
            return -1
        
        alice_points = 0
        bob_points = total_points
        
        for i in range(n):
            alice_points += possible[i]
            bob_points -= possible[i]
            
            # Alice must play at least one level
            if i >= 0 and alice_points > bob_points:
                return i + 1
        
        return -1