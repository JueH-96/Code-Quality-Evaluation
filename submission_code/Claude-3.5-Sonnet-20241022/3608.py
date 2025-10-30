class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        def gcd(a: int, b: int) -> int:
            while b:
                a, b = b, a % b
            return a
        
        def calculate_gcd_subsequences(mask: int) -> int:
            # Calculate GCD of elements in the subsequence represented by mask
            curr_gcd = 0
            for i in range(n):
                if mask & (1 << i):
                    if curr_gcd == 0:
                        curr_gcd = nums[i]
                    else:
                        curr_gcd = gcd(curr_gcd, nums[i])
            return curr_gcd
        
        # Store subsequences grouped by their GCD
        gcd_groups = {}
        
        # Generate all possible subsequences using bitmask
        for mask in range(1, 1 << n):
            curr_gcd = calculate_gcd_subsequences(mask)
            if curr_gcd not in gcd_groups:
                gcd_groups[curr_gcd] = []
            gcd_groups[curr_gcd].append(mask)
        
        result = 0
        
        # For each GCD value, count pairs of disjoint subsequences
        for gcd_value in gcd_groups:
            subsequences = gcd_groups[gcd_value]
            length = len(subsequences)
            
            # Check each pair of subsequences
            for i in range(length):
                for j in range(i, length):
                    mask1 = subsequences[i]
                    mask2 = subsequences[j]
                    
                    # Check if subsequences are disjoint
                    if (mask1 & mask2) == 0:
                        # Add 2 if different subsequences, add 1 if same subsequence
                        if i != j:
                            result = (result + 2) % MOD
                        else:
                            result = (result + 1) % MOD
        
        return result