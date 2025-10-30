class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        max_bit = 30  # since 2^30 is about 1e9
        cnt = [0] * (max_bit + 1)
        for num in nums:
            for b in range(max_bit + 1):
                if num & (1 << b):
                    cnt[b] += 1
        
        part1 = 0
        for b in range(max_bit + 1):
            part1 += cnt[b] * ((1 << b) ** 2)
        
        part2 = 0
        for i in range(max_bit + 1):
            for j in range(i + 1, max_bit + 1):
                a = cnt[i]
                b_cnt = cnt[j]
                min_ab = min(a, b_cnt)
                part2 += 2 * min_ab * ((1 << i) * (1 << j))
        
        total = (part1 + part2) % MOD
        return total