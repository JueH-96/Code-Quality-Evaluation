class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = float('inf')
        prev = set()
        for i in range(n):
            base = nums[i]
            curr = set()
            curr.add(base)
            diff0 = abs(k - base)
            if diff0 == 0:
                return 0
            if diff0 < ans:
                ans = diff0
            for val in prev:
                new_val = val | base
                curr.add(new_val)
            for val in curr:
                diff = abs(k - val)
                if diff == 0:
                    return 0
                if diff < ans:
                    ans = diff
            prev = curr
        return ans