class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        
        # Calculate the total possible points for Alice and Bob
        total_points = sum(possible)
        
        # If all levels are impossible, Alice cannot gain more points than Bob
        if total_points == 0:
            return -1
        
        # Alice's points and Bob's points
        alice_points = 0
        bob_points = total_points
        
        # Alice must play at least one level
        for i in range(n):
            # Alice plays the i-th level
            if possible[i] == 1:
                alice_points += 1
            else:
                alice_points -= 1
            
            # Bob's remaining points
            if possible[i] == 1:
                bob_points -= 1
            else:
                bob_points += 1
            
            # Check if Alice has more points than Bob
            if alice_points > bob_points:
                return i + 1
        
        # If we reach here, Alice cannot have more points than Bob
        return -1