class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        n = len(s)
        if k == 0:
            return True
        
        # Precompute first and last occurrence for each character (0-25 for a-z)
        first_occurrence = [-1] * 26
        last_occurrence = [-1] * 26
        
        for i in range(n):
            c = s[i]
            idx = ord(c) - ord('a')
            if first_occurrence[idx] == -1:
                first_occurrence[idx] = i
            last_occurrence[idx] = i
        
        # Generate all candidate intervals (L, R) where L is first of some char, R last of some char
        candidates = set()
        for c in range(26):
            L = first_occurrence[c]
            if L == -1:
                continue
            for d in range(26):
                R = last_occurrence[d]
                if R == -1:
                    continue
                if L <= R:
                    candidates.add((L, R))
        
        # Check each candidate for validity
        valid_intervals = []
        for (L, R) in candidates:
            if (L, R) == (0, n-1):
                continue  # exclude the entire string
            
            valid = True
            for c in range(26):
                first_c = first_occurrence[c]
                last_c = last_occurrence[c]
                if first_c == -1 or last_c == -1:
                    continue  # character not present in s
                # Check if the character is present in [L, R]
                if (first_c <= R) and (last_c >= L):
                    # Check if first_c >= L and last_c <= R
                    if first_c < L or last_c > R:
                        valid = False
                        break
            if valid:
                valid_intervals.append((L, R))
        
        # If no valid intervals, return k ==0 (but k is at least 1 here)
        if not valid_intervals:
            return False
        
        # Sort intervals by their end to apply greedy algorithm
        valid_intervals.sort(key=lambda x: x[1])
        
        count = 0
        last_end = -1
        for interval in valid_intervals:
            current_L, current_R = interval
            if current_L > last_end:
                count += 1
                last_end = current_R
                if count >= k:
                    break  # early exit if possible
        
        return count >= k