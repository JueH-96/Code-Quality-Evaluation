class Solution:
    def beautifulSplits(self, nums):
        n = len(nums)
        if n < 3:
            return 0
        MOD1 = 10**9 + 7
        MOD2 = 10**9 + 9
        base = 131
        H1 = [0] * (n + 1)
        H2 = [0] * (n + 1)
        pow1 = [1] * (n + 1)
        pow2 = [1] * (n + 1)
        for i in range(1, n + 1):
            H1[i] = (H1[i-1] * base + nums[i-1]) % MOD1
            H2[i] = (H2[i-1] * base + nums[i-1]) % MOD2
            pow1[i] = (pow1[i-1] * base) % MOD1
            pow2[i] = (pow2[i-1] * base) % MOD2
        
        def get_double_hash(l, r):
            len_seg = r - l
            hash1 = (H1[r] - H1[l] * pow1[len_seg]) % MOD1
            hash2 = (H2[r] - H2[l] * pow2[len_seg]) % MOD2
            if hash1 < 0:
                hash1 += MOD1
            if hash2 < 0:
                hash2 += MOD2
            return (hash1, hash2)
        
        countA = 0
        for i in range(1, (n - 1) // 2 + 1):
            if get_double_hash(0, i) == get_double_hash(i, 2 * i):
                countA += (n - 2 * i)
        
        countB = 0
        for j in range(2, n):
            maxL = min(j - 1, n - j)
            for L in range(1, maxL + 1):
                i_val = j - L
                if get_double_hash(i_val, j) == get_double_hash(j, j + L):
                    countB += 1
        
        countBoth = 0
        for i in range(1, (n // 3) + 1):
            if get_double_hash(0, i) == get_double_hash(i, 2 * i):
                start_j = 2 * i
                end_j = min(n - 1, (n + i) // 2)
                if start_j > end_j:
                    continue
                for j in range(start_j, end_j + 1):
                    if get_double_hash(i, j) == get_double_hash(j, 2 * j - i):
                        countBoth += 1
        
        return countA + countB - countBoth