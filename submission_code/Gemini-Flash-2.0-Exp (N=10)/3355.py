class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        
        for alice_levels in range(1, n):
            alice_score = 0
            for i in range(alice_levels):
                if possible[i] == 1:
                    alice_score += 1
                else:
                    alice_score -= 1
            
            bob_score = 0
            for i in range(alice_levels, n):
                if possible[i] == 1:
                    bob_score += 1
                else:
                    bob_score -= 1
            
            if alice_score > bob_score:
                return alice_levels
        
        return -1