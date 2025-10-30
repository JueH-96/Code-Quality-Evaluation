class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        # Step 1: Count set bits in each bit position
        bit_counts = [0] * 31
        for num in nums:
            for p in range(31):
                if num & (1 << p):
                    bit_counts[p] += 1
        
        # Step 2: Initialize k elements to zero
        elements = [0] * k
        
        # Step 3: Assign set bits to elements
        for p in range(30, -1, -1):
            c_p = bit_counts[p]
            if c_p == 0:
                continue
            # Assign set bits to elements starting from the first
            for i in range(k):
                if c_p == 0:
                    break
                elements[i] |= (1 << p)
                c_p -= 1
                if c_p == 0:
                    break
        
        # Step 4: Calculate sum of squares modulo 10^9 + 7
        sum_squares = 0
        for x in elements:
            sum_squares = (sum_squares + (x * x)) % MOD
        return sum_squares