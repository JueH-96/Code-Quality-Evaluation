class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Quick check: if total sum of nums is smaller than target,
        # it's impossible to form target as a subsequence (even after breaking).
        if sum(nums) < target:
            return -1
        
        # Count how many of each power of two we have (since nums[i] is a power of two).
        # We'll only need to consider powers up to 2^30 (constraint).
        freq = [0]*31
        for x in nums:
            p = x.bit_length() - 1  # because x = 2^p
            freq[p] += 1
        
        operations = 0
        
        # We'll go bit by bit from lowest (0) to highest (30).
        # For each bit i that is set in 'target', we need one 2^i. If we don't already
        # have a 2^i in freq[i], we'll break a bigger power down to get it.
        for i in range(31):
            # Check if bit i of target is set:
            if (target >> i) & 1:
                # We need one 2^i
                if freq[i] > 0:
                    # We already have it; just use it
                    freq[i] -= 1
                else:
                    # We must break down a bigger power 2^j, j>i
                    found = False
                    for j in range(i+1, 31):
                        if freq[j] > 0:
                            # Take one 2^j and break it down fully to 2^(j-i) copies of 2^i
                            freq[j] -= 1
                            freq[i] += (1 << (j - i))  # 2^(j-i)
                            operations += (j - i)      # Need j-i breaks
                            found = True
                            break
                    if not found:
                        return -1  # Cannot produce the needed 2^i
                    # Now use one of the newly formed 2^i
                    freq[i] -= 1
            
            # After possibly using one 2^i for the current bit,
            # any leftover freq[i] can stay as is. We do NOT gain
            # any operation to merge them upward (that operation isn't allowed).
        
        return operations