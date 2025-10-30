class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        if k == 0:
            return True
        n = len(s)
        if n == 0:
            return False
        
        # Precompute first and last occurrences of each character
        first = {}
        last = {}
        for i, c in enumerate(s):
            if c not in first:
                first[c] = i
            last[c] = i
        
        # Precompute first_occurrence and last_occurrence arrays
        first_occurrence = [0] * n
        last_occurrence = [0] * n
        for i in range(n):
            c = s[i]
            first_occurrence[i] = first[c]
            last_occurrence[i] = last[c]
        
        # Precompute log table for range queries
        log_table = [0] * (n + 1)
        for i in range(2, n + 1):
            log_table[i] = log_table[i // 2] + 1
        
        # Build sparse tables for min and max queries on first_occurrence and last_occurrence
        max_log = log_table[n] + 1 if n > 0 else 0
        st_min = [[0] * max_log for _ in range(n)]
        st_max = [[0] * max_log for _ in range(n)]
        
        for i in range(n):
            st_min[i][0] = first_occurrence[i]
            st_max[i][0] = last_occurrence[i]
        
        for j in range(1, max_log):
            for i in range(n - (1 << j) + 1):
                st_min[i][j] = min(st_min[i][j-1], st_min[i + (1 << (j-1))][j-1])
                st_max[i][j] = max(st_max[i][j-1], st_max[i + (1 << (j-1))][j-1])
        
        # Functions to query min and max in range [L, R]
        def get_min(L, R):
            if L > R:
                return float('inf')
            length = R - L + 1
            k = log_table[length]
            return min(st_min[L][k], st_min[R - (1 << k) + 1][k])
        
        def get_max(L, R):
            if L > R:
                return -1
            length = R - L + 1
            k = log_table[length]
            return max(st_max[L][k], st_max[R - (1 << k) + 1][k])
        
        # Generate all candidate intervals from pairs of first[c] and last[d]
        candidates = set()
        present_chars = list(first.keys())
        for c in present_chars:
            for d in present_chars:
                L = first[c]
                R = last[d]
                if L > R:
                    continue
                if L == 0 and R == n - 1:
                    continue
                min_first = get_min(L, R)
                max_last = get_max(L, R)
                if min_first >= L and max_last <= R:
                    candidates.add((L, R))
        
        if not candidates:
            return False
        
        # Sort candidates by their end position
        sorted_candidates = sorted(candidates, key=lambda x: x[1])
        
        # Greedy interval selection
        count = 0
        last_end = -1
        for (start, end) in sorted_candidates:
            if start > last_end:
                count += 1
                last_end = end
                if count >= k:
                    return True
        
        return count >= k