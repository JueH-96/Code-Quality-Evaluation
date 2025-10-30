class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Precompute GCD for all possible subsets
        gcd_groups = collections.defaultdict(list)
        for mask in range(1, 1 << n):
            subset = [nums[i] for i in range(n) if mask & (1 << i)]
            gcd_val = subset[0]
            for num in subset[1:]:
                gcd_val = math.gcd(gcd_val, num)
            gcd_groups[gcd_val].append(mask)
        
        # Function to check if two masks are disjoint
        def are_disjoint(mask1, mask2):
            return (mask1 & mask2) == 0
        
        # Count valid pairs
        total = 0
        for gcd_val in gcd_groups:
            masks = gcd_groups[gcd_val]
            count = 0
            for i in range(len(masks)):
                for j in range(i + 1, len(masks)):
                    if are_disjoint(masks[i], masks[j]):
                        count = (count + 1) % MOD
            # Since pairs are unordered, multiply by 2
            total = (total + 2 * count) % MOD
        
        return total