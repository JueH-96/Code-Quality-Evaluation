class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Quick check: if total sum of nums is less than target, it's impossible.
        s = sum(nums)
        if s < target:
            return -1
        
        # Count how many of each power of two we have, up to 2^30.
        # (Because nums[i] <= 2^30 per the problem statement.)
        MAX_POW = 30
        cnt = [0]*(MAX_POW+1)
        for val in nums:
            p = val.bit_length() - 1  # because val is a power of two
            cnt[p] += 1
        
        # We'll also extract the bits needed from target.
        # target < 2^31, so we check powers up to 30 (0-based).
        # We'll process from largest down to 0.
        cost = 0
        
        # Function to find a bigger power to break down if needed.
        def break_from_bigger(k):
            # look for any j>k where cnt[j] > 0
            # break exactly one 2^j into 2^(j-k) pieces of 2^k
            nonlocal cost
            for j in range(k+1, MAX_POW+1):
                if cnt[j] > 0:
                    cnt[j] -= 1
                    # cost to split 2^j down to 2^k is (j-k) operations
                    cost += (j - k)
                    # produce 2^(j-k) pieces of 2^k
                    return 2**(j-k)
            return 0  # no bigger power found => cannot break
        
        # We'll go from the highest bit to the lowest bit of the target.
        for k in range(MAX_POW, -1, -1):
            # Check if this bit is set in target
            bit_needed = (target >> k) & 1
            
            # We need bit_needed many of 2^k (so 0 or 1 in normal binary),
            # but possibly we didn't manage to get a leftover from a bigger break,
            # so let's see if we have enough in cnt[k].
            if bit_needed == 1:
                if cnt[k] == 0:
                    # Try to break from bigger powers
                    made = break_from_bigger(k)
                    if made == 0:
                        return -1  # impossible to fulfill this bit
                    # we use 1 piece out of made, leftover below:
                    cnt[k] += (made - 1)  # newly created pieces of 2^k
                else:
                    # We already have enough to fulfill the one bit we need
                    pass
                
                # now fulfill this bit from our supply
                if cnt[k] <= 0:
                    return -1  # couldn't fulfill it
                cnt[k] -= 1  # used one 2^k for the target bit
            
            # Now any leftover 2^k in cnt[k] can be broken into 2^(k-1)
            # to help with lower bits if k > 0.
            if k > 0:
                leftover = cnt[k]
                if leftover > 0:
                    # Breaking each leftover 2^k once costs 1 operation each,
                    # producing two 2^(k-1).
                    cost += leftover
                    cnt[k-1] += leftover * 2
                    cnt[k] = 0
        
        # If we get here, we've successfully formed all bits of target.
        return cost