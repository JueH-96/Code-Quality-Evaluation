import bisect

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        if n < 2:
            return []
        
        max_num = max(nums)
        freq = [0] * (max_num + 1)
        for num in nums:
            freq[num] += 1
        
        # Compute the count of numbers divisible by each d
        cnt = [0] * (max_num + 1)
        for d in range(1, max_num + 1):
            for multiple in range(d, max_num + 1, d):
                cnt[d] += freq[multiple]
        
        # Calculate total pairs for each d
        total_pairs = [0] * (max_num + 1)
        for d in range(1, max_num + 1):
            c = cnt[d]
            total_pairs[d] = c * (c - 1) // 2
        
        # Calculate exact pairs using inclusion-exclusion principle
        exact_pairs = [0] * (max_num + 1)
        for d in range(max_num, 0, -1):
            exact_pairs[d] = total_pairs[d]
            k = 2 * d
            while k <= max_num:
                exact_pairs[d] -= exact_pairs[k]
                k += d
        
        # Build the prefix sum array
        prefix_list = [0]
        current_sum = 0
        for d in range(1, max_num + 1):
            current_sum += exact_pairs[d]
            prefix_list.append(current_sum)
        
        # Process each query using binary search
        ans = []
        for q in queries:
            idx = bisect.bisect_right(prefix_list, q)
            ans.append(idx)
        
        return ans