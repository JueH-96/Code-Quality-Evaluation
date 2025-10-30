class Solution:
    def minimumSteps(self, s: str) -> int:
        count_zeros = 0
        inversions = 0
        n = len(s)
        
        # Iterate from the end to the beginning
        for i in range(n-1, -1, -1):
            if s[i] == '0':
                count_zeros += 1
            else:
                inversions += count_zeros
                
        return inversions