class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        count = 0
        n = len(word)
        for i in range(n):
            for j in range(i, n):
                substring = word[i:j+1]
                char_counts = {}
                valid = True
                for char in substring:
                    char_counts[char] = char_counts.get(char, 0) + 1
                
                for char in char_counts:
                    if char_counts[char] != k:
                        valid = False
                        break
                
                if not valid:
                    continue

                for l in range(len(substring) - 1):
                    c1 = substring[l]
                    c2 = substring[l+1]
                    if abs(ord(c1) - ord(c2)) > 2:
                        valid = False
                        break
                if valid:
                    count +=1

        return count