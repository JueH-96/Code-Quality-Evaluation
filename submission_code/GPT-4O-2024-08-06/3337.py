class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        # Find all the indices where character c appears in the string s
        indices = [i for i, char in enumerate(s) if char == c]
        
        # If there are less than 2 occurrences of c, no valid substrings can be formed
        if len(indices) < 2:
            return len(indices)  # If there's only one occurrence, it counts as one substring
        
        # Calculate the number of substrings
        count = 0
        n = len(indices)
        
        # For each pair of indices (i, j) where i <= j, we have a valid substring
        for i in range(n):
            for j in range(i, n):
                count += 1
        
        return count