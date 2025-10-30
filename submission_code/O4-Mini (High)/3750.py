class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        from collections import defaultdict
        import bisect

        n = len(nums)
        # Map each value to the sorted list of its occurrence indices
        pos = defaultdict(list)
        for i, v in enumerate(nums):
            pos[v].append(i)
        for v in pos:
            pos[v].sort()

        ans = []
        for qi in queries:
            v = nums[qi]
            arr = pos[v]
            # If this value occurs only once, no other index matches
            if len(arr) == 1:
                ans.append(-1)
            else:
                # Find the position of qi in the sorted occurrences list
                idx = bisect.bisect_left(arr, qi)
                # Left neighbor in the circular list
                left = arr[idx-1] if idx > 0 else arr[-1]
                # Right neighbor in the circular list
                right = arr[idx+1] if idx+1 < len(arr) else arr[0]
                # Compute direct distances
                d1 = abs(qi - left)
                d2 = abs(right - qi)
                # In a circular array, you can go the other way around too
                ans.append(min(d1, n - d1, d2, n - d2))
        return ans