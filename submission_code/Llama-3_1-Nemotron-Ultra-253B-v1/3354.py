class Solution:
    def minimizeStringValue(self, s: str) -> str:
        n = len(s)
        # Precompute suffix_fixed array to store counts of fixed characters from each position
        suffix_fixed = [[0] * 26 for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(26):
                suffix_fixed[i][j] = suffix_fixed[i + 1][j]
            if s[i] != '?':
                c = s[i]
                idx = ord(c) - ord('a')
                suffix_fixed[i][idx] += 1
        
        current_count = [0] * 26
        res = []
        for i in range(n):
            if s[i] != '?':
                c = s[i]
                idx = ord(c) - ord('a')
                current_count[idx] += 1
                res.append(c)
            else:
                min_val = None
                best_c = None
                for c_ord in range(26):
                    # Calculate the value as current_count + future fixed occurrences
                    val = current_count[c_ord] + suffix_fixed[i + 1][c_ord]
                    if min_val is None or val < min_val or (val == min_val and c_ord < best_c):
                        min_val = val
                        best_c = c_ord
                chosen_char = chr(best_c + ord('a'))
                current_count[best_c] += 1
                res.append(chosen_char)
        return ''.join(res)