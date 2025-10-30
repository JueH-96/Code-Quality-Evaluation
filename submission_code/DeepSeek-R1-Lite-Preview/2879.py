class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        
        # Precompute divisors for each length from 1 to n
        divisors_len = {}
        for length in range(1, n + 1):
            divisors = []
            for d in range(1, length):
                if length % d == 0:
                    divisors.append(d)
            divisors_len[length] = divisors
        
        # Precompute cost_table[i][j]: minimum changes for s[i..j] to be a semi-palindrome
        cost_table = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                length = j - i + 1
                min_changes = float('inf')
                for d in divisors_len[length]:
                    total_changes = 0
                    for r in range(d):
                        positions = range(i + r, j + 1, d)
                        seq = [s[p] for p in positions]
                        changes = 0
                        m = len(seq)
                        for a in range(m // 2):
                            if seq[a] != seq[m - 1 - a]:
                                changes += 1
                        total_changes += changes
                    min_changes = min(min_changes, total_changes)
                cost_table[i][j] = min_changes
        
        # Initialize DP table
        DP = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        DP[0][0] = 0
        
        # Fill DP table
        for i in range(1, n + 1):
            for current_k in range(1, k + 1):
                if current_k > i // 2:
                    continue  # Not possible to have current_k partitions for i characters
                for m in range(current_k - 1, i):
                    if DP[m][current_k - 1] + cost_table[m][i - 1] < DP[i][current_k]:
                        DP[i][current_k] = DP[m][current_k - 1] + cost_table[m][i - 1]
        
        return DP[n][k]