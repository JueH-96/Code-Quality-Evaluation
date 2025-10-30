class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        if n == 1:
            return True

        def check(arr):
            s = sum(arr)
            if len(arr) == 1 or s >= m:
                return True
            return False

        def solve(arr):
            if len(arr) == n:
                return True
            
            for i in range(1, len(arr)):
                left = arr[:i]
                right = arr[i:]
                if check(left) and check(right):
                    if solve(left + right):
                        return True
            return False

        return solve(nums)