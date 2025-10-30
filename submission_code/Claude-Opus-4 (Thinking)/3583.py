class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        max_val = max(nums)
        
        # Count frequency of each number
        freq = [0] * (max_val + 1)
        for num in nums:
            freq[num] += 1
        
        # Count pairs for each divisor
        divisor_pairs = [0] * (max_val + 1)
        for d in range(1, max_val + 1):
            count = 0
            for multiple in range(d, max_val + 1, d):
                count += freq[multiple]
            if count >= 2:
                divisor_pairs[d] = count * (count - 1) // 2
        
        # Calculate exact GCD counts using inclusion-exclusion
        gcd_count = [0] * (max_val + 1)
        for g in range(max_val, 0, -1):
            gcd_count[g] = divisor_pairs[g]
            for multiple in range(2 * g, max_val + 1, g):
                gcd_count[g] -= gcd_count[multiple]
        
        # Answer queries
        result = []
        for q in queries:
            # Find the q-th element (0-indexed)
            idx = 0
            for g in range(1, max_val + 1):
                if idx + gcd_count[g] > q:
                    result.append(g)
                    break
                idx += gcd_count[g]
        
        return result