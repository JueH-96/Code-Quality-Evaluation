class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        from math import gcd
        import bisect
        max_num = max(nums)
        n = len(nums)
        freq = [0] * (max_num + 1)
        for num in nums:
            freq[num] += 1
        
        total_multiples = [0] * (max_num + 1)
        for g in range(1, max_num + 1):
            for m in range(g, max_num + 1, g):
                total_multiples[g] += freq[m]
        
        total_pairs = [0] * (max_num + 1)
        for g in range(1, max_num + 1):
            total_pairs[g] = total_multiples[g] * (total_multiples[g] - 1) // 2
        
        count_pairs = [0] * (max_num + 1)
        for g in range(max_num, 0, -1):
            count_pairs[g] = total_pairs[g]
            for m in range(2 * g, max_num + 1, g):
                count_pairs[g] -= count_pairs[m]
        
        gcd_list = []
        for g in range(1, max_num + 1):
            if count_pairs[g] > 0:
                gcd_list.append((g, count_pairs[g]))
        gcd_list.sort()
        
        prefix_counts = []
        total = 0
        for g, cnt in gcd_list:
            total += cnt
            prefix_counts.append((total, g))
        
        result = []
        for q in queries:
            idx = bisect.bisect_left(prefix_counts, (q + 1, 0))
            if idx < len(prefix_counts):
                result.append(prefix_counts[idx][1])
            else:
                # This should not happen as queries are within the range
                result.append(-1)
        return result