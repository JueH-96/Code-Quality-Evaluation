class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        from collections import defaultdict
        mod = 10 ** 9 + 7
        n = len(nums)
        
        # Precompute factorials and inverse factorials
        max_n = n + 1
        factorial = [1] * max_n
        inverse = [1] * max_n
        inv = [1] * max_n
        for i in range(2, max_n):
            factorial[i] = factorial[i - 1] * i % mod
            inv[i] = mod - mod // i * inv[mod % i] % mod
            inverse[i] = inverse[i - 1] * inv[i] % mod
        
        def nCr(n, r):
            if n < r or r < 0:
                return 0
            return factorial[n] * inverse[r] % mod * inverse[n - r] % mod

        ans = 0
        nums_len = len(nums)
        for i in range(2, nums_len - 2):
            counts_before = defaultdict(int)
            counts_after = defaultdict(int)
            total_before = i
            total_after = nums_len - i - 1
            for idx in range(i):
                counts_before[nums[idx]] += 1
            for idx in range(i + 1, nums_len):
                counts_after[nums[idx]] += 1
            
            counts_before_total = i
            counts_after_total = nums_len - i - 1
            nums_i = nums[i]

            counts_before_nums_i = counts_before[nums_i]
            counts_after_nums_i = counts_after[nums_i]
            
            for k1 in range(3):
                for k2 in range(3):
                    if k1 + k2 < 2:
                        continue
                    if counts_before_nums_i < k1 or counts_after_nums_i < k2:
                        continue
                    ways_before = nCr(counts_before_nums_i, k1) * nCr(counts_before_total - counts_before_nums_i, 2 - k1) % mod
                    ways_after = nCr(counts_after_nums_i, k2) * nCr(counts_after_total - counts_after_nums_i, 2 - k2) % mod

                    freq_nums_i = 1 + k1 + k2
                    
                    # Now check if any other element can match or exceed freq_nums_i
                    possible = True
                    max_freq_possible = freq_nums_i - 1  # They cannot reach freq_nums_i
                    for x in set(counts_before.keys()).union(counts_after.keys()):
                        if x == nums_i:
                            continue
                        freq_x = min(counts_before[x], 2 - k1) + min(counts_after[x], 2 - k2)
                        if freq_x >= freq_nums_i:
                            possible = False
                            break
                    if possible:
                        total_ways = ways_before * ways_after % mod
                        ans = (ans + total_ways) % mod
        return ans