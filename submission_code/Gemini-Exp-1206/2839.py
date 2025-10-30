class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        nums = sorted(zip(nums1, nums2), key=lambda x: -x[0])
        q = sorted([(x, y, i) for i, (x, y) in enumerate(queries)], key=lambda x: -x[0])
        ans = [-1] * len(queries)
        stack = []
        j = 0
        for x, y, i in q:
            while j < n and nums[j][0] >= x:
                a, b = nums[j]
                while stack and stack[-1][1] <= a + b:
                    stack.pop()
                if not stack or stack[-1][0] < b:
                    stack.append((b, a + b))
                j += 1
            l, r = 0, len(stack) - 1
            while l <= r:
                mid = (l + r) // 2
                if stack[mid][0] >= y:
                    ans[i] = stack[mid][1]
                    r = mid - 1
                else:
                    l = mid + 1
        return ans