class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        dp = {}
        
        for i in range(n):
            dp[(i, 1 << i)] = 1
            
        for mask_bits_count in range(2, n + 1):
            for mask in range(1, 1 << n):
                if bin(mask).count('1') == mask_bits_count:
                    for j in range(n):
                        if (mask >> j) & 1:
                            current_dp_value = 0
                            prev_mask = mask ^ (1 << j)
                            for i in range(n):
                                if i != j and ((prev_mask >> i) & 1):
                                    if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
                                        current_dp_value = (current_dp_value + dp.get((i, prev_mask), 0)) % MOD
                            if current_dp_value > 0:
                                dp[(j, mask)] = current_dp_value
                                
        total_special_permutations = 0
        final_mask = (1 << n) - 1
        for i in range(n):
            total_special_permutations = (total_special_permutations + dp.get((i, final_mask), 0)) % MOD
            
        return total_special_permutations