class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        n_half = n // 2
        
        # Precompute prefix sums for each character
        prefix = [[0] * (n + 1) for _ in range(26)]
        for i in range(n):
            current_char = s[i]
            current_code = ord(current_char) - ord('a')
            for c in range(26):
                prefix[c][i + 1] = prefix[c][i]
            prefix[current_code][i + 1] += 1
        
        # Compute even_total
        counts = [0] * 26
        for c in s:
            counts[ord(c) - ord('a')] += 1
        even_total = all(v % 2 == 0 for v in counts)
        
        result = []
        for a, b, c, d in queries:
            if not even_total:
                result.append(False)
                continue
            
            # Compute F1 frequencies
            f1 = [0] * 26
            for i in range(26):
                part1 = prefix[i][a]
                part2 = prefix[i][n_half] - prefix[i][b + 1]
                f1[i] = part1 + part2
            
            # Compute F2 frequencies
            f2 = [0] * 26
            for i in range(26):
                part1 = prefix[i][c] - prefix[i][n_half]
                part2 = prefix[i][n] - prefix[i][d + 1]
                f2[i] = part1 + part2
            
            # Compute A_freq
            a_freq = [0] * 26
            for i in range(26):
                a_freq[i] = prefix[i][b + 1] - prefix[i][a]
            
            # Compute B_freq
            b_freq = [0] * 26
            for i in range(26):
                b_freq[i] = prefix[i][d + 1] - prefix[i][c]
            
            # Check condition for each character
            possible = True
            for i in range(26):
                if f1[i] + a_freq[i] != f2[i] + b_freq[i]:
                    possible = False
                    break
            result.append(possible)
        
        return result