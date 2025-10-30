class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        # Precompute prefix sum array
        prefix = [[0] * 26 for _ in range(n + 1)]
        for j in range(n):
            for c in range(26):
                prefix[j + 1][c] = prefix[j][c]
            current_char = ord(s[j]) - ord('a')
            prefix[j + 1][current_char] += 1
        
        total = 0
        
        for i in range(n):
            min_j = n  # Initialize to a value larger than possible
            for c in range(26):
                # Binary search for j_plus_1 in [i, n]
                low = i
                high = n
                res = -1
                while low <= high:
                    mid = (low + high) // 2
                    cnt = prefix[mid][c] - prefix[i][c]
                    if cnt >= k:
                        res = mid
                        high = mid - 1
                    else:
                        low = mid + 1
                if res != -1:
                    j_candidate = res - 1
                    if j_candidate >= i and j_candidate < min_j:
                        min_j = j_candidate
            if min_j != n:
                total += (n - min_j)
        
        return total