class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        if sum(possible) < 2:
            return -1
        
        alice_points = 0
        bob_points = sum(possible)  # Bob will play all possible levels
        
        for i in range(n):
            if possible[i] == 1:
                alice_points += 1  # Alice clears the level
                bob_points -= 1  # Bob loses the level
            else:
                alice_points -= 1  # Alice fails the level
                bob_points += 1  # Bob clears the level
            
            # Check if Alice has more points than Bob
            if alice_points > bob_points:
                return i + 1  # Return the number of levels Alice has played
        
        return -1  # If Alice cannot gain more points than Bob