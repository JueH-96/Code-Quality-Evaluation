class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Count the frequency of each power of 2 in nums
        from collections import defaultdict
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        # Decompose the target into its binary representation
        # to find the required powers of 2
        required = defaultdict(int)
        current = target
        power = 0
        while current > 0:
            if current & 1:
                required[1 << power] += 1
            current >>= 1
            power += 1
        
        # Now, we need to match the required powers with the available ones
        # and perform operations to split larger powers into smaller ones
        operations = 0
        for p in sorted(required, reverse=True):
            if freq[p] >= required[p]:
                continue
            needed = required[p] - freq[p]
            # We need to split higher powers to get the required p
            # Start from the next higher power and split down
            higher_p = p * 2
            while higher_p <= max(freq.keys()):
                if freq[higher_p] > 0:
                    # Split one higher_p into two p's
                    freq[higher_p] -= 1
                    freq[p] += 2
                    operations += 1
                    needed -= 1
                    if needed <= 0:
                        break
                higher_p *= 2
            if needed > 0:
                # If we still need more p's, but no higher powers are available
                return -1
        
        return operations