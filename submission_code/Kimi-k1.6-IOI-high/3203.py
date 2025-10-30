class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        m = n // 2
        left_str = s[:m]
        right_str = s[m:]
        
        # Precompute prefix sums for each character in left and right halves
        left_pre = [[0] * (m + 1) for _ in range(26)]
        for i in range(m):
            for char_code in range(26):
                left_pre[char_code][i+1] = left_pre[char_code][i] + (1 if left_str[i] == chr(ord('a') + char_code) else 0)
        
        right_pre = [[0] * (m + 1) for _ in range(26)]
        for i in range(m):
            for char_code in range(26):
                right_pre[char_code][i+1] = right_pre[char_code][i] + (1 if right_str[i] == chr(ord('a') + char_code) else 0)
        
        # Precompute mismatch array and its prefix sum
        mismatch = [0] * m
        for i in range(m):
            if left_str[i] != right_str[m - 1 - i]:
                mismatch[i] = 1
        
        prefix_mismatch = [0] * (m + 1)
        for i in range(m):
            prefix_mismatch[i + 1] = prefix_mismatch[i] + mismatch[i]
        total_mismatch = prefix_mismatch[m]
        
        result = []
        for a, b, c, d in queries:
            # Calculate mirrored intervals in the left half for the right rearranged region
            C = (2 * m - 1) - d
            D = (2 * m - 1) - c
            
            # Check condition 1: fixed regions must already be palindromic
            left_start = min(a, C)
            left_end = max(b, D)
            overlap_left = (a <= D) and (C <= b)
            if overlap_left:
                sum_ab_cd = prefix_mismatch[left_end + 1] - prefix_mismatch[left_start]
            else:
                sum_ab = prefix_mismatch[b + 1] - prefix_mismatch[a]
                sum_cd = prefix_mismatch[D + 1] - prefix_mismatch[C]
                sum_ab_cd = sum_ab + sum_cd
            
            sum_forbidden = total_mismatch - sum_ab_cd
            if sum_forbidden != 0:
                result.append(False)
                continue
            
            # Check condition 2: character counts in left_mod and right_mod must match
            valid = True
            c_right = c - m
            d_right = d - m
            L = (2 * m - 1) - b
            R = (2 * m - 1) - a
            L_right = L - m
            R_right = R - m
            
            for char_code in range(26):
                # Calculate left_count
                if overlap_left:
                    lc = left_pre[char_code][left_end + 1] - left_pre[char_code][left_start]
                else:
                    lc_ab = left_pre[char_code][b + 1] - left_pre[char_code][a]
                    lc_cd = left_pre[char_code][D + 1] - left_pre[char_code][C]
                    lc = lc_ab + lc_cd
                
                # Calculate right_count
                right_start = min(c_right, L_right)
                right_end = max(d_right, R_right)
                overlap_right = (c_right <= R_right) and (L_right <= d_right)
                if overlap_right:
                    rc = right_pre[char_code][right_end + 1] - right_pre[char_code][right_start]
                else:
                    rc_ab = right_pre[char_code][d_right + 1] - right_pre[char_code][c_right]
                    rc_cd = right_pre[char_code][R_right + 1] - right_pre[char_code][L_right]
                    rc = rc_ab + rc_cd
                
                if lc != rc:
                    valid = False
                    break
            
            result.append(valid)
        
        return result