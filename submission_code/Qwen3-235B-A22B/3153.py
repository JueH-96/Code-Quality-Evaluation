class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        bit_count = [0] * 31
        for num in nums:
            for i in range(31):
                if num & (1 << i):
                    bit_count[i] += 1
        
        pow2 = [1] * 31
        for i in range(1, 31):
            pow2[i] = (pow2[i-1] * 2) % MOD
        
        s = [min(bit_count[i], k) for i in range(31)]
        
        total = 0
        
        # Handle the first part: sum of squares of individual bits
        for i in range(31):
            term = (pow2[i] * pow2[i]) % MOD
            term = (term * s[i]) % MOD
            total = (total + term) % MOD
        
        # Handle the second part: cross terms
        for i in range(31):
            for j in range(i + 1, 31):
                m = min(s[i], s[j])
                term = (pow2[i] * pow2[j]) % MOD
                term = (term * m) % MOD
                term = (term * 2) % MOD
                total = (total + term) % MOD
        
        return total