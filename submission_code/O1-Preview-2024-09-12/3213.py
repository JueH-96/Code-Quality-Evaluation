class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        left = [ -1 ] * n
        right = [ n ] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                left[i] = stack[-1]
            stack.append(i)
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()
            if stack:
                right[i] = stack[-1]
            stack.append(i)
        
        from collections import defaultdict
        intervals = defaultdict(list)
        for i in range(n):
            m = nums[i]
            l = left[i] + 1
            r = right[i] -1
            intervals[m].append((l, r))
        
        result = 0
        for m in intervals:
            # Merge overlapping intervals
            intervals_m = sorted(intervals[m])
            merged = []
            for start, end in intervals_m:
                if not merged or start > merged[-1][1]:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
            # For each merged interval, process
            for l, r in merged:
                arr = [1 if nums[i]==m else 0 for i in range(l, r+1)]
                cumsum = [0]
                for x in arr:
                    cumsum.append(cumsum[-1]+x)
                import bisect
                prefix_sums = []
                for i in range(len(cumsum)):
                    prefix_sums.append(cumsum[i])
                total = 0
                from bisect import bisect_left, insort
                sorted_prefix = []
                for i in range(len(cumsum)-1, -1, -1):
                    curr_sum = cumsum[i]
                    target = curr_sum + k
                    idx = bisect_left(sorted_prefix, target)
                    total += len(sorted_prefix) - idx
                    insort(sorted_prefix, curr_sum)
                result += total
        return result