from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        if not nums:
            return []
        
        max_num = max(nums)
        n = len(nums)
        
        # Step 1: Compute count array
        count = [0] * (max_num + 1)
        for x in nums:
            # Find all divisors of x
            for d in range(1, int(x**0.5) + 1):
                if x % d == 0:
                    count[d] += 1
                    other = x // d
                    if other != d:
                        count[other] += 1
        
        # Step 2: Compute total_pairs for each d
        total_pairs = [0] * (max_num + 1)
        for d in range(1, max_num + 1):
            c = count[d]
            total_pairs[d] = c * (c - 1) // 2
        
        # Step 3: Compute exact array
        exact = [0] * (max_num + 1)
        for d in range(max_num, 0, -1):
            exact[d] = total_pairs[d]
            m = 2 * d
            while m <= max_num:
                exact[d] -= exact[m]
                m += d
        
        # Step 4: Compute prefix array
        prefix = [0] * (max_num + 1)
        for d in range(1, max_num + 1):
            prefix[d] = prefix[d - 1] + exact[d]
        
        # Process each query
        result = []
        for q in queries:
            low, high = 1, max_num
            ans = max_num
            while low <= high:
                mid = (low + high) // 2
                if prefix[mid] > q:
                    ans = mid
                    high = mid - 1
                else:
                    low = mid + 1
            result.append(ans)
        
        return result