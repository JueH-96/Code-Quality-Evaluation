class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        
        # Convert to points: 1 -> +1, 0 -> -1
        points = [1 if x == 1 else -1 for x in possible]
        
        # Calculate total sum
        total_sum = sum(points)
        
        # Try each possible split point
        alice_sum = 0
        
        # Alice must play at least 1 level, Bob must play at least 1 level
        # So Alice can play from 1 to n-1 levels
        for i in range(1, n):
            alice_sum += points[i-1]  # Add the (i-1)th level to Alice's score
            bob_sum = total_sum - alice_sum
            
            if alice_sum > bob_sum:
                return i
        
        return -1