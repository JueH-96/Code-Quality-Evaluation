class Solution:
    def minimumLevels(self, possible: list[int]) -> int:
        n = len(possible)
        
        # Calculate the total score if Bob plays all levels
        total_score = sum(possible) - n
        
        # Alice's and Bob's scores
        alice_score = 0
        bob_score = total_score
        
        # Iterate through the levels to find the minimum levels Alice should play
        for i in range(n):
            # Update Alice's and Bob's scores
            alice_score += 1 if possible[i] == 1 else -1
            bob_score += 1 if possible[i] == 0 else -1
            
            # Check if Alice has more points than Bob
            if alice_score > bob_score:
                return i + 1
            
            # If this is the last level, and Alice hasn't gained more points, it's not possible
            if i == n - 2:
                return -1
        
        # If we reach here, it means Alice can never gain more points than Bob
        return -1