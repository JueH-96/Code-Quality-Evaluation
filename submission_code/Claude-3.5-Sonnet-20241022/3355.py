class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        
        # For each prefix length i, calculate Alice's and Bob's scores
        for i in range(1, n):
            alice_score = sum(1 if x == 1 else -1 for x in possible[:i])
            bob_score = sum(1 if x == 1 else -1 for x in possible[i:])
            
            # If Alice's score is greater than Bob's score, return the prefix length
            if alice_score > bob_score:
                return i
                
        return -1