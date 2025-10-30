class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        import sys
        n = len(nums)

        # Count how many elements have each bit set
        # We'll use up to 60 bits to be safe, though 32 or so might suffice
        freq = [0]*60
        for x in nums:
            b = 0
            while x > 0:
                if x & 1:
                    freq[b] += 1
                x >>= 1
                b += 1
        
        # Precompute prefix OR and suffix OR
        prefixOR = [0]*n
        suffixOR = [0]*n

        prefixOR[0] = nums[0]
        for i in range(1, n):
            prefixOR[i] = prefixOR[i-1] | nums[i]

        suffixOR[n-1] = nums[n-1]
        for i in range(n-2, -1, -1):
            suffixOR[i] = suffixOR[i+1] | nums[i]

        # Function to get OR of array excluding index i
        def exclude_or(i: int) -> int:
            if i == 0:
                return suffixOR[1] if n > 1 else 0
            elif i == n-1:
                return prefixOR[n-2] if n > 1 else 0
            else:
                return prefixOR[i-1] | suffixOR[i+1]

        ans = 0
        for i in range(n):
            base = exclude_or(i)
            x = nums[i]

            # Determine which bits are uniquely contributed by nums[i]
            # i.e., bits that appear exactly once in the entire array and are in nums[i]
            uniq_bits_mask = 0
            tmp = x
            b = 0
            while tmp > 0:
                if tmp & 1:
                    if freq[b] == 1:  # this bit is uniquely contributed by x
                        uniq_bits_mask |= (1 << b)
                tmp >>= 1
                b += 1

            # We can shift the uniquely contributed bits by k positions
            # The rest of the bits in x stay where they are
            # new_x = (bits we shift) << k | (bits we do not shift)
            shifted = ((x & uniq_bits_mask) << k) | (x & ~uniq_bits_mask)

            candidate = base | shifted
            ans = max(ans, candidate)

        return ans