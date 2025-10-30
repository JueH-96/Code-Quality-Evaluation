class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2
        
        cnt_left = [0] * 26
        cnt_right = [0] * 26
        for i in range(half):
            cnt_left[ord(s[i]) - ord('a')] += 1
        for i in range(half, n):
            cnt_right[ord(s[i]) - ord('a')] += 1
        if cnt_left != cnt_right:
            return [False] * len(queries)
        
        global_pref = [[0] * 26 for _ in range(n+1)]
        for i in range(n):
            char_index = ord(s[i]) - ord('a')
            global_pref[i+1] = global_pref[i][:]
            global_pref[i+1][char_index] += 1
        
        left_error = [0] * half
        for i in range(half):
            j = n - 1 - i
            if s[i] != s[j]:
                left_error[i] = 1
        left_error_prefix = [0] * (half + 1)
        for i in range(half):
            left_error_prefix[i+1] = left_error_prefix[i] + left_error[i]
        
        def get_freq(l, r):
            res = [0] * 26
            if l > r:
                return res
            for idx in range(26):
                res[idx] = global_pref[r+1][idx] - global_pref[l][idx]
            return res
        
        def get_error(l, r):
            if l > r:
                return 0
            return left_error_prefix[r+1] - left_error_prefix[l]
        
        ans = []
        for a, b, c, d in queries:
            L1 = n - 1 - d
            R1 = n - 1 - c
            segs_R2 = []
            seg_low1 = max(0, L1)
            seg_high1 = min(a - 1, R1)
            if seg_low1 <= seg_high1:
                segs_R2.append((seg_low1, seg_high1))
            seg_low2 = max(b + 1, L1)
            seg_high2 = min(half - 1, R1)
            if seg_low2 <= seg_high2:
                segs_R2.append((seg_low2, seg_high2))
            
            R2_vec = [0] * 26
            for seg in segs_R2:
                l_seg, r_seg = seg
                freq = get_freq(l_seg, r_seg)
                for idx in range(26):
                    R2_vec[idx] += freq[idx]
            
            L2 = n - 1 - b
            R2_val = n - 1 - a
            segs_R3 = []
            seg_low3 = max(half, L2)
            seg_high3 = min(c - 1, R2_val)
            if seg_low3 <= seg_high3:
                segs_R3.append((seg_low3, seg_high3))
            seg_low4 = max(d + 1, L2)
            seg_high4 = min(n - 1, R2_val)
            if seg_low4 <= seg_high4:
                segs_R3.append((seg_low4, seg_high4))
            
            R3_vec = [0] * 26
            for seg in segs_R3:
                l_seg, r_seg = seg
                freq = get_freq(l_seg, r_seg)
                for idx in range(26):
                    R3_vec[idx] += freq[idx]
            
            total_error = 0
            if a > 0:
                total_error += get_error(0, a - 1)
            if b + 1 <= half - 1:
                total_error += get_error(b + 1, half - 1)
            for seg in segs_R2:
                l_seg, r_seg = seg
                total_error -= get_error(l_seg, r_seg)
            if total_error != 0:
                ans.append(False)
                continue
            
            A_vec = get_freq(a, b)
            B_vec = get_freq(c, d)
            valid = True
            for i in range(26):
                if R2_vec[i] > B_vec[i] or R3_vec[i] > A_vec[i]:
                    valid = False
                    break
                if A_vec[i] - R3_vec[i] != B_vec[i] - R2_vec[i]:
                    valid = False
                    break
            ans.append(valid)
        
        return ans