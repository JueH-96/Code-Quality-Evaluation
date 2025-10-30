import bisect

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        if not nums:
            return []
        
        max_num = max(nums)
        freq = [0] * (max_num + 1)
        for x in nums:
            freq[x] += 1
        
        # Compute cnt[d] for each d
        cnt = [0] * (max_num + 1)
        for d in range(1, max_num + 1):
            for multiple in range(d, max_num + 1, d):
                cnt[d] += freq[multiple]
        
        # Compute total_pairs[d]
        total_pairs = [0] * (max_num + 1)
        for d in range(max_num, 0, -1):
            total_pairs[d] = (cnt[d] * (cnt[d] - 1)) // 2
            m = 2 * d
            while m <= max_num:
                total_pairs[d] -= total_pairs[m]
                m += d
        
        # Collect sorted_pairs
        sorted_pairs = []
        for d in range(1, max_num + 1):
            if total_pairs[d] > 0:
                sorted_pairs.append((d, total_pairs[d]))
        
        # Compute starts
        starts = [0]
        current = 0
        for d, f in sorted_pairs:
            current += f
            starts.append(current)
        
        # Process queries
        result = []
        for q in queries:
            idx = bisect.bisect_right(starts, q) - 1
            result.append(sorted_pairs[idx][0])
        
        return result