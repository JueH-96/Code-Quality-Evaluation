class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        alice_score = 0
        bob_score = 0
        
        for i in range(n - 1):
            if possible[i] == 1:
                alice_score += 1
            else:
                alice_score -= 1
            
            for j in range(i + 1, n):
                if possible[j] == 1:
                    bob_score += 1
                else:
                    bob_score -= 1
            
            if alice_score > bob_score:
                return i + 1
            
            bob_score = 0
        
        return -1