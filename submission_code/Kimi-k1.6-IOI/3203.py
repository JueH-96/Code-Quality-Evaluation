class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        m = n // 2
        L = s[:m]
        R = s[m:]
        R_rev = R[::-1]
        
        # Compute diff array and prefix_diff
        diff = [0] * m
        for i in range(m):
            if L[i] != R_rev[i]:
                diff[i] = 1
        prefix_diff = [0] * (m + 1)
        for i in range(m):
            prefix_diff[i+1] = prefix_diff[i] + diff[i]
        
        # Build frequency arrays for L and R_rev
        freq_L = [[0] for _ in range(26)]
        freq_R = [[0] for _ in range(26)]
        for i in range(m):
            char_l = L[i]
            char_r = R_rev[i]
            for c in range(26):
                freq_L[c].append(freq_L[c][i] + (1 if chr(ord('a') + c) == char_l else 0))
                freq_R[c].append(freq_R[c][i] + (1 if chr(ord('a') + c) == char_r else 0))
        
        answer = []
        for query in queries:
            a, b, c, d = query
            mc = n - 1 - d
            md = n - 1 - c
            
            overlap_start = max(a, mc)
            overlap_end = min(b, md)
            
            sum_a_b = prefix_diff[b+1] - prefix_diff[a]
            sum_mc_md = prefix_diff[md+1] - prefix_diff[mc]
            
            sum_intersect = (prefix_diff[overlap_end + 1] - prefix_diff[overlap_start]) if overlap_start <= overlap_end else 0
            sum_union = sum_a_b + sum_mc_md - sum_intersect
            
            total_diff = prefix_diff[m]
            if total_diff - sum_union != 0:
                answer.append(False)
                continue
            
            valid = True
            for c_char in range(26):
                # Calculate count in L's union
                count_l = (freq_L[c_char][b+1] - freq_L[c_char][a]) + (freq_L[c_char][md+1] - freq_L[c_char][mc])
                if overlap_start <= overlap_end:
                    count_l -= (freq_L[c_char][overlap_end + 1] - freq_L[c_char][overlap_start])
                
                # Calculate count in R_rev's union
                count_r = (freq_R[c_char][b+1] - freq_R[c_char][a]) + (freq_R[c_char][md+1] - freq_R[c_char][mc])
                if overlap_start <= overlap_end:
                    count_r -= (freq_R[c_char][overlap_end + 1] - freq_R[c_char][overlap_start])
                
                if count_l != count_r:
                    valid = False
                    break
            
            answer.append(valid)
        
        return answer