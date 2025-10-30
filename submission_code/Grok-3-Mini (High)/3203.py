import bisect
from typing import List

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        mid = n // 2
        
        # Precompute bad_k where s[k] != s[n-1-k]
        bad_k = [k for k in range(mid) if s[k] != s[n-1-k]]
        
        # Precompute prefix sums for each character
        prefix_sum = [[0] * (n + 1) for _ in range(26)]
        for i in range(n):
            char_idx = ord(s[i]) - ord('a')
            for c in range(26):
                prefix_sum[c][i + 1] = prefix_sum[c][i]
                if c == char_idx:
                    prefix_sum[c][i + 1] += 1  # Increment count for the correct character
        
        ans = []
        
        for query in queries:
            a, b, c, d = query
            
            # Step 1: Difference check to ensure all mismatched pairs are covered
            # Intervals for k: [a, b] and [n-1-d, n-1-c]
            start1, end1 = a, b
            start2, end2 = n - 1 - d, n - 1 - c
            
            # Sort intervals by start
            if start1 > start2:
                start1, end1, start2, end2 = start2, end2, start1, end1  # Swap
            
            cov_min_k = start1
            cov_max_k = max(end1, end2)
            
            # Check if any bad_k is outside the union
            if bad_k and bad_k[0] < cov_min_k:
                ans.append(False)
                continue
            if bad_k and bad_k[-1] > cov_max_k:
                ans.append(False)
                continue
            
            # Check if intervals are disjoint and handle the gap
            if end1 < start2:  # Disjoint intervals
                gap_low = end1 + 1
                gap_high = start2 - 1
                if gap_low <= gap_high:
                    idx = bisect.bisect_left(bad_k, gap_low)
                    if idx < len(bad_k) and bad_k[idx] <= gap_high:
                        ans.append(False)
                        continue
            
            # Difference check passed, now multiset checks
            
            # Left side demanded frequency
            demanded_freq_left = [0] * 26
            X_d_val = n - 1 - d
            Y_c_val = n - 1 - c
            
            # First demanded k interval for left
            low_k1 = a
            high_k1 = min(b, X_d_val - 1)
            if low_k1 <= high_k1 and low_k1 >= 0 and high_k1 <= mid - 1:
                start_m1 = n - 1 - high_k1
                end_m1 = n - 1 - low_k1
                if 0 <= start_m1 <= end_m1 < n:
                    for char_idx in range(26):
                        freq_c = prefix_sum[char_idx][end_m1 + 1] - prefix_sum[char_idx][start_m1]
                        demanded_freq_left[char_idx] += freq_c
            
            # Second demanded k interval for left
            low_k2 = max(a, Y_c_val + 1)
            high_k2 = b
            if low_k2 <= high_k2 and low_k2 >= 0 and high_k2 <= mid - 1:
                start_m2 = n - 1 - high_k2
                end_m2 = n - 1 - low_k2
                if 0 <= start_m2 <= end_m2 < n:
                    for char_idx in range(26):
                        freq_c = prefix_sum[char_idx][end_m2 + 1] - prefix_sum[char_idx][start_m2]
                        demanded_freq_left[char_idx] += freq_c
            
            # Frequency in left substring s[a..b]
            freq_left = [prefix_sum[char_idx][b + 1] - prefix_sum[char_idx][a] for char_idx in range(26)]
            
            # Check if frequency meets demand for left
            if any(freq_left[idx] < demanded_freq_left[idx] for idx in range(26)):
                ans.append(False)
                continue
            
            # Right side demanded frequency
            demanded_freq_right = [0] * 26
            
            # First demanded k interval for right
            low_k_right1 = n - 1 - d
            high_k_right1 = min(n - 1 - c, a - 1)
            if low_k_right1 <= high_k_right1 and low_k_right1 >= 0 and high_k_right1 <= mid - 1:
                for char_idx in range(26):
                    freq_c = prefix_sum[char_idx][high_k_right1 + 1] - prefix_sum[char_idx][low_k_right1]
                    demanded_freq_right[char_idx] += freq_c
            
            # Second demanded k interval for right
            low_k_right2 = max(n - 1 - d, b + 1)
            high_k_right2 = min(n - 1 - c, mid - 1)
            if low_k_right2 <= high_k_right2 and low_k_right2 >= 0 and high_k_right2 <= mid - 1:
                for char_idx in range(26):
                    freq_c = prefix_sum[char_idx][high_k_right2 + 1] - prefix_sum[char_idx][low_k_right2]
                    demanded_freq_right[char_idx] += freq_c
            
            # Frequency in right substring s[c..d]
            freq_right_sub = [prefix_sum[char_idx][d + 1] - prefix_sum[char_idx][c] for char_idx in range(26)]
            
            # Check if frequency meets demand for right
            if any(freq_right_sub[idx] < demanded_freq_right[idx] for idx in range(26)):
                ans.append(False)
                continue
            
            # Compute remaining frequencies
            rem_freq_left = [freq_left[idx] - demanded_freq_left[idx] for idx in range(26)]
            rem_freq_right = [freq_right_sub[idx] - demanded_freq_right[idx] for idx in range(26)]
            
            # Check if remaining multisets are equal
            if rem_freq_left == rem_freq_right:
                ans.append(True)
            else:
                ans.append(False)
        
        return ans