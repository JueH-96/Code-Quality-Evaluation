class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        for i in range(n):
            for j in range(i, n):
                substring = s[i:j+1]
                char_counts = {}
                for char in substring:
                    char_counts[char] = char_counts.get(char, 0) + 1
                
                found_k = False
                for char in char_counts:
                    if char_counts[char] >= k:
                        found_k = True
                        break
                if found_k:
                    count += 1
        return count