class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)

        def check(t):
            current_values = sorted([nums1[i] + t * nums2[i] for i in range(n)])
            for k in range(n + 1):
                if sum(current_values[:k]) <= x:
                    return True
            return False

        low = 0
        high = 2 * 10**6  # A reasonable upper bound
        ans = -1

        while low <= high:
            mid = (low + high) // 2
            if check(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1

        return ans