class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        M = 60
        cnt = [0] * (M + 1)
        
        for x in nums:
            exp = x.bit_length() - 1
            if exp <= M:
                cnt[exp] += 1
        
        bits = [0] * (M + 1)
        t = target
        for i in range(M + 1):
            bits[i] = t & 1
            t //= 2
        
        operations = 0
        for i in range(M + 1):
            if bits[i] == 1:
                if cnt[i] > 0:
                    cnt[i] -= 1
                else:
                    j = i + 1
                    while j <= M and cnt[j] == 0:
                        j += 1
                    if j > M:
                        return -1
                    operations += (j - i)
                    cnt[j] -= 1
                    for k in range(i, j):
                        cnt[k] += 1
                    cnt[i] -= 1
            if i < M:
                cnt[i + 1] += cnt[i] // 2
                cnt[i] = cnt[i] % 2
        
        return operations