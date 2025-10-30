class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        n = len(s)
        if k == 0:
            return True
        
        first = [-1] * 26
        last = [-1] * 26
        
        for i in range(n):
            c = s[i]
            idx = ord(c) - ord('a')
            if first[idx] == -1:
                first[idx] = i
            last[idx] = i
        
        # Generate candidate intervals
        candidates = set()
        for c in range(26):
            if first[c] == -1:
                continue
            for d in range(26):
                if last[d] == -1:
                    continue
                i_val = first[c]
                j_val = last[d]
                if i_val <= j_val and (j_val - i_val + 1) < n:
                    candidates.add((i_val, j_val))
        
        # Check validity of each candidate interval
        valid_intervals = []
        for (i, j) in candidates:
            chars_in_sub = set()
            for k_idx in range(i, j + 1):
                chars_in_sub.add(s[k_idx])
            valid = True
            for ch in chars_in_sub:
                idx_ch = ord(ch) - ord('a')
                if first[idx_ch] < i or last[idx_ch] > j:
                    valid = False
                    break
            if valid:
                valid_intervals.append((i, j))
        
        # Sort intervals by their end points
        valid_intervals.sort(key=lambda x: x[1])
        
        # Greedy selection of non-overlapping intervals
        count = 0
        current_end = -1
        for interval in valid_intervals:
            start, end = interval
            if start > current_end:
                count += 1
                current_end = end
        
        return count >= k