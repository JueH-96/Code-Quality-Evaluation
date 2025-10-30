class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
        
        mod1 = 10**9 + 7
        mod2 = 10**9 + 9
        base1 = 131
        base2 = 137
        
        h1 = [0] * n
        h2 = [0] * n
        p1 = [1] * n
        p2 = [1] * n
        
        h1[0] = nums[0] % mod1
        h2[0] = nums[0] % mod2
        for i in range(1, n):
            h1[i] = (h1[i-1] * base1 + nums[i]) % mod1
            h2[i] = (h2[i-1] * base2 + nums[i]) % mod2
            p1[i] = (p1[i-1] * base1) % mod1
            p2[i] = (p2[i-1] * base2) % mod2
        
        countA = 0
        max_i = (n - 1) // 2
        for i in range(1, max_i + 1):
            if i == 0:
                hash1_left = h1[i-1]
                hash2_left = h2[i-1]
            else:
                hash1_left = h1[i-1]
                hash2_left = h2[i-1]
            
            l = i
            r = 2 * i - 1
            if l == 0:
                hash1_right = h1[r]
                hash2_right = h2[r]
            else:
                seg_len = r - l + 1
                hash1_right = (h1[r] - h1[l-1] * p1[seg_len]) % mod1
                hash2_right = (h2[r] - h2[l-1] * p2[seg_len]) % mod2
                if hash1_right < 0:
                    hash1_right += mod1
                if hash2_right < 0:
                    hash2_right += mod2
            
            if hash1_left == hash1_right and hash2_left == hash2_right:
                countA += (n - 2 * i)
        
        countB = 0
        for j in range(1, n):
            maxL = min(j - 1, n - j)
            if maxL < 1:
                continue
            for L in range(1, maxL + 1):
                left_l = j - L
                left_r = j - 1
                if left_l == 0:
                    hash1_left_seg = h1[left_r]
                    hash2_left_seg = h2[left_r]
                else:
                    seg_len = L
                    hash1_left_seg = (h1[left_r] - h1[left_l - 1] * p1[seg_len]) % mod1
                    hash2_left_seg = (h2[left_r] - h2[left_l - 1] * p2[seg_len]) % mod2
                    if hash1_left_seg < 0:
                        hash1_left_seg += mod1
                    if hash2_left_seg < 0:
                        hash2_left_seg += mod2
                
                right_l = j
                right_r = j + L - 1
                seg_len = L
                hash1_right_seg = (h1[right_r] - h1[right_l - 1] * p1[seg_len]) % mod1
                hash2_right_seg = (h2[right_r] - h2[right_l - 1] * p2[seg_len]) % mod2
                if hash1_right_seg < 0:
                    hash1_right_seg += mod1
                if hash2_right_seg < 0:
                    hash2_right_seg += mod2
                
                if hash1_left_seg == hash1_right_seg and hash2_left_seg == hash2_right_seg:
                    countB += 1
        
        countAB = 0
        for i in range(1, max_i + 1):
            if i == 0:
                hash1_left1 = h1[i-1]
                hash2_left1 = h2[i-1]
            else:
                hash1_left1 = h1[i-1]
                hash2_left1 = h2[i-1]
            
            l = i
            r = 2 * i - 1
            if l == 0:
                hash1_right1 = h1[r]
                hash2_right1 = h2[r]
            else:
                seg_len = r - l + 1
                hash1_right1 = (h1[r] - h1[l-1] * p1[seg_len]) % mod1
                hash2_right1 = (h2[r] - h2[l-1] * p2[seg_len]) % mod2
                if hash1_right1 < 0:
                    hash1_right1 += mod1
                if hash2_right1 < 0:
                    hash2_right1 += mod2
            
            if not (hash1_left1 == hash1_right1 and hash2_left1 == hash2_right1):
                continue
            
            j_min = 2 * i
            j_max = min(n - 1, (n + i) // 2)
            if j_min > j_max:
                continue
            for j in range(j_min, j_max + 1):
                L_val = j - i
                left_l_ab = i
                left_r_ab = j - 1
                seg_len_ab = left_r_ab - left_l_ab + 1
                hash1_left_ab = (h1[left_r_ab] - h1[left_l_ab - 1] * p1[seg_len_ab]) % mod1
                hash2_left_ab = (h2[left_r_ab] - h2[left_l_ab - 1] * p2[seg_len_ab]) % mod2
                if hash1_left_ab < 0:
                    hash1_left_ab += mod1
                if hash2_left_ab < 0:
                    hash2_left_ab += mod2
                
                right_l_ab = j
                right_r_ab = j + L_val - 1
                seg_len_ab = L_val
                hash1_right_ab = (h1[right_r_ab] - h1[right_l_ab - 1] * p1[seg_len_ab]) % mod1
                hash2_right_ab = (h2[right_r_ab] - h2[right_l_ab - 1] * p2[seg_len_ab]) % mod2
                if hash1_right_ab < 0:
                    hash1_right_ab += mod1
                if hash2_right_ab < 0:
                    hash2_right_ab += mod2
                
                if hash1_left_ab == hash1_right_ab and hash2_left_ab == hash2_right_ab:
                    countAB += 1
        
        return countA + countB - countAB