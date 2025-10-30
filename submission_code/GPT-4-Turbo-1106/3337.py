class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        count = 0
        occurrences = [i for i, char in enumerate(s) if char == c]
        
        for i in range(len(occurrences)):
            for j in range(i, len(occurrences)):
                count += (occurrences[j] - occurrences[i] + 1)
                
        return count