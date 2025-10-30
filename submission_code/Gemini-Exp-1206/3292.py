class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n, m = len(nums), len(changeIndices)

        def check(t):
            last_pos = {}
            for i in range(t - 1, -1, -1):
                if changeIndices[i] not in last_pos:
                    last_pos[changeIndices[i]] = i
            marked = [False] * n
            extra = 0
            needed = 0
            for i in range(t):
                idx = changeIndices[i] - 1
                if i == last_pos.get(changeIndices[i], -1) and not marked[idx]:
                    marked[idx] = True
                    needed += nums[idx]
                else:
                    extra += 1
            return extra >= needed and all(marked)

        if len(set(changeIndices)) < n:
            return -1

        left, right = 1, m
        ans = -1
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        return ans