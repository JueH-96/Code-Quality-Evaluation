from typing import List

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        # use 64-bit rolling hash
        MASK = (1 << 64) - 1
        BASE = 131
        
        # prefix hashes H and powers P
        H = [0] * (n + 1)
        P = [1] * (n + 1)
        for i in range(n):
            H[i+1] = ((H[i] * BASE) + (nums[i] + 1)) & MASK
            P[i+1] = (P[i] * BASE) & MASK
        
        def get_hash(l: int, r: int) -> int:
            # hash of nums[l:r]
            return (H[r] - (H[l] * P[r-l] & MASK)) & MASK
        
        # cond1_prefix[i] = True if nums[0:i] == nums[i:2*i]
        cond1_prefix = [False] * (n + 1)
        count1 = 0
        # i must be at least 1, and 2*i <= n-1+1 = n
        # but for non-empty nums3 we need 2*i <= n-1 => i <= (n-1)//2
        for i in range(1, (n-1)//2 + 1):
            if get_hash(0, i) == get_hash(i, 2*i):
                cond1_prefix[i] = True
                # valid j are from 2*i to n-1 inclusive, count = (n-1) - 2*i + 1 = n - 2*i
                count1 += (n - 2*i)
        
        # count splits where nums2 is prefix of nums3 (cond2)
        # and count overlaps where both cond1 and cond2 hold
        count2 = 0
        overlaps = 0
        
        # i from 1 to n-2 (to leave space for nums2 and nums3 non-empty)
        for i in range(1, n-1):
            # for cond2 we need j-i <= n-j  => j <= (n+i)//2
            j_max = (n + i) // 2
            if j_max > n-1:
                j_max = n-1
            # iterate j from i+1 to j_max
            for j in range(i+1, j_max+1):
                L = j - i
                # check nums[i:j] == nums[j:j+L]
                if get_hash(i, j) == get_hash(j, j + L):
                    count2 += 1
                    # if also cond1 holds for this i and j>=2*i
                    if cond1_prefix[i] and j >= 2*i:
                        overlaps += 1
        
        # total = cond1 only + cond2 only + both => count1 + count2 - overlaps
        return count1 + count2 - overlaps