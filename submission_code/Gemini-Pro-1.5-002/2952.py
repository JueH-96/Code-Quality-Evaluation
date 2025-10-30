class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        pairs = sorted(zip(nums2, nums1))
        nums2, nums1 = zip(*pairs)
        nums1 = list(nums1)
        nums2 = list(nums2)

        def check(t):
            total_sum = sum(num1 + num2 * t for num1, num2 in zip(nums1, nums2))
            
            for i in range(1 << n):
                current_sum = total_sum
                for j in range(n):
                    if (i >> j) & 1:
                        current_sum -= nums1[j] + nums2[j] * t
                if current_sum <= x:
                    return True
            return False

        left, right = 0, 2 * 10**6  # Adjust the upper bound as needed
        ans = -1

        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return ans