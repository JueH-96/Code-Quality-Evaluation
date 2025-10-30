import bisect

class Solution:
    def maxDistinctElements(self, nums: list[int], k: int) -> int:
        if not nums:
            return 0
        
        d = len(set(nums))
        freq = {}
        for x in nums:
            freq[x] = freq.get(x, 0) + 1
        
        duplicates = []
        for x, cnt in freq.items():
            for _ in range(cnt - 1):
                duplicates.append((x - k, x + k))
        
        duplicates.sort(key=lambda x: x[1])
        
        distinct_set_sorted = sorted(freq.keys())
        cur = -10**18
        count = 0
        
        for l, r in duplicates:
            candidate = max(cur, l)
            idx = bisect.bisect_left(distinct_set_sorted, candidate)
            while idx < len(distinct_set_sorted) and distinct_set_sorted[idx] == candidate:
                candidate += 1
                idx += 1
            if candidate <= r:
                count += 1
                cur = candidate + 1
        
        return d + count