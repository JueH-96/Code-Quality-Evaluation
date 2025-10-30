class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        total_substrings = n * (n + 1) // 2
        non_dominant_count = 0
        
        # Iterate over all possible substrings
        for i in range(n):
            zeros = 0
            ones = 0
            for j in range(i, n):
                if s[j] == '0':
                    zeros += 1
                else:
                    ones += 1
                
                # Check if the substring is non-dominant
                if ones < zeros * zeros:
                    non_dominant_count += 1
        
        # Dominant substrings are total substrings minus non-dominant substrings
        dominant_count = total_substrings - non_dominant_count
        return dominant_count