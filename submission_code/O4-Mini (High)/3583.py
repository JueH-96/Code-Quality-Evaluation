from typing import List
import bisect

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        # Maximum value in nums
        maxA = max(nums)
        
        # Frequency of each number in nums
        freq = [0] * (maxA + 1)
        for x in nums:
            freq[x] += 1
        
        # cnt[d] = number of pairs (i<j) where both nums[i] and nums[j] are divisible by d
        cnt = [0] * (maxA + 1)
        for d in range(1, maxA + 1):
            c = 0
            # sum up how many numbers are multiples of d
            for m in range(d, maxA + 1, d):
                c += freq[m]
            if c >= 2:
                cnt[d] = c * (c - 1) // 2
        
        # exact[d] = number of pairs whose gcd is exactly d
        # use inclusion–exclusion (Mobius inversion style)
        exact = [0] * (maxA + 1)
        for d in range(maxA, 0, -1):
            s = cnt[d]
            # subtract counts of multiples of d
            k = 2 * d
            while k <= maxA:
                s -= exact[k]
                k += d
            exact[d] = s
        
        # prefix[d] = total number of gcd‐pairs with gcd <= d
        prefix = [0] * (maxA + 1)
        for d in range(1, maxA + 1):
            prefix[d] = prefix[d - 1] + exact[d]
        
        # Answer each query by binary searching for the smallest d
        # such that prefix[d] > query_index, i.e. prefix[d] >= query_index+1
        ans = []
        for q in queries:
            idx = bisect.bisect_left(prefix, q + 1)
            ans.append(idx)
        
        return ans