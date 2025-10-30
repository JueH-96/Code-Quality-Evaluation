class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        
        # Calculate the maximum points Alice can get
        alice_max = sum(possible)
        
        # Calculate the maximum points Bob can get
        bob_max = 0
        for i in range(n):
            if possible[i] == 1:
                bob_max += 1
            else:
                bob_max -= 1
        
        # If Alice can't get more points than Bob, return -1
        if alice_max <= bob_max:
            return -1
        
        # Find the minimum number of levels Alice should play
        alice_points = 0
        for i in range(n):
            if possible[i] == 1:
                alice_points += 1
            else:
                alice_points -= 1
            
            if alice_points > bob_max:
                return i + 1
        
        return n