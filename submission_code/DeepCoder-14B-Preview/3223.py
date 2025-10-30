class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        if n == 0 or k == 0:
            return 0
        
        # Precompute the valid array
        valid = [False] * (n - 1)
        for i in range(n - 1):
            c1 = ord(word[i]) - ord('a')
            c2 = ord(word[i + 1]) - ord('a')
            if abs(c1 - c2) <= 2:
                valid[i] = True
            else:
                valid[i] = False
        
        # Precompute max_reach array
        max_reach = [0] * n
        max_reach[-1] = n - 1
        for i in range(n - 2, -1, -1):
            if valid[i]:
                max_reach[i] = max_reach[i + 1]
            else:
                max_reach[i] = i
        
        # Precompute prefix sums for each character
        prefix = [[0] * 26]
        for i in range(n):
            new_row = prefix[i].copy()
            char = ord(word[i]) - ord('a')
            new_row[char] += 1
            prefix.append(new_row)
        
        count = 0
        
        for i in range(n):
            end = max_reach[i]
            for j in range(i, end + 1):
                L = j - i + 1
                if L % k != 0:
                    continue
                
                valid_sub = True
                m = 0
                for c in range(26):
                    cnt = prefix[j + 1][c] - prefix[i][c]
                    if cnt == 0:
                        continue
                    elif cnt != k:
                        valid_sub = False
                        break
                    else:
                        m += 1
                
                if valid_sub:
                    if m * k == L:
                        count += 1
        
        return count