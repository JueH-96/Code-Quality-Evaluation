class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        mid = n // 2
        
        E = [0] * mid
        for i in range(mid):
            if s[i] != s[n-1-i]:
                E[i] = 1
        
        P = [0] * mid
        if mid > 0:
            P[0] = E[0]
            for i in range(1, mid):
                P[i] = P[i-1] + E[i]
        
        prefix1 = [[0]*26 for _ in range(mid)]
        if mid > 0:
            c0 = s[0]
            idx0 = ord(c0) - ord('a')
            prefix1[0][idx0] = 1
            for i in range(1, mid):
                prev = prefix1[i-1][:]
                c = s[i]
                idx = ord(c) - ord('a')
                prev[idx] += 1
                prefix1[i] = prev
        
        prefix2 = [[0]*26 for _ in range(n - mid)]
        if n - mid > 0:
            c0 = s[mid]
            idx0 = ord(c0) - ord('a')
            prefix2[0][idx0] = 1
            for i in range(mid+1, n):
                pos = i - mid
                c = s[i]
                idx = ord(c) - ord('a')
                if pos > 0:
                    new_arr = prefix2[pos-1][:]
                    new_arr[idx] += 1
                    prefix2[pos] = new_arr
                else:
                    prefix2[0][idx] += 1
        
        def get_sum_E(l, r):
            if l > r:
                return 0
            if l == 0:
                return P[r]
            else:
                return P[r] - P[l-1]
        
        def get_freq1(l, r):
            if l > r:
                return [0]*26
            if l == 0:
                return prefix1[r][:]
            else:
                res = [prefix1[r][i] - prefix1[l-1][i] for i in range(26)]
                return res
        
        def get_freq2(l, r):
            if l > r:
                return [0]*26
            l2 = l - mid
            r2 = r - mid
            if l2 < 0 or r2 < 0:
                return [0]*26
            if l2 == 0:
                return prefix2[r2][:]
            else:
                res = [prefix2[r2][i] - prefix2[l2-1][i] for i in range(26)]
                return res
        
        ans = []
        for q in queries:
            a, b, c, d = q
            total_error = 0
            I1_low = 0
            I1_high = min(a-1, n-d-2)
            total_error += get_sum_E(I1_low, I1_high)
            
            I2_low = max(0, n-c)
            I2_high = min(a-1, mid-1)
            total_error += get_sum_E(I2_low, I2_high)
            
            I3_low = b+1
            I3_high = min(mid-1, n-d-2)
            total_error += get_sum_E(I3_low, I3_high)
            
            I4_low = max(b+1, n-c)
            I4_high = mid-1
            total_error += get_sum_E(I4_low, I4_high)
            
            if total_error > 0:
                ans.append(False)
                continue
            
            j1_low = max(mid, n-1-b)
            j1_high = min(c-1, n-1-a)
            j2_low = max(d+1, n-1-b)
            j2_high = min(n-1, n-1-a)
            
            forced_A = [0]*26
            if j1_low <= j1_high:
                freq_temp = get_freq2(j1_low, j1_high)
                for idx in range(26):
                    forced_A[idx] += freq_temp[idx]
            if j2_low <= j2_high:
                freq_temp = get_freq2(j2_low, j2_high)
                for idx in range(26):
                    forced_A[idx] += freq_temp[idx]
            
            i1_low = max(0, n-1-d)
            i1_high = min(a-1, n-1-c)
            i2_low = max(b+1, n-1-d)
            i2_high = min(mid-1, n-1-c)
            
            forced_B = [0]*26
            if i1_low <= i1_high:
                freq_temp = get_freq1(i1_low, i1_high)
                for idx in range(26):
                    forced_B[idx] += freq_temp[idx]
            if i2_low <= i2_high:
                freq_temp = get_freq1(i2_low, i2_high)
                for idx in range(26):
                    forced_B[idx] += freq_temp[idx]
            
            freq_left = get_freq1(a, b)
            freq_right = get_freq2(c, d)
            
            possible = True
            for char_idx in range(26):
                needed_left = forced_A[char_idx]
                if freq_left[char_idx] < needed_left:
                    possible = False
                    break
                freq_left[char_idx] -= needed_left
                
                needed_right = forced_B[char_idx]
                if freq_right[char_idx] < needed_right:
                    possible = False
                    break
                freq_right[char_idx] -= needed_right
            
            if not possible:
                ans.append(False)
                continue
            
            for char_idx in range(26):
                if (freq_left[char_idx] % 2) != (freq_right[char_idx] % 2):
                    possible = False
                    break
                if freq_left[char_idx] != freq_right[char_idx]:
                    possible = False
                    break
                    
            ans.append(possible)
        
        return ans