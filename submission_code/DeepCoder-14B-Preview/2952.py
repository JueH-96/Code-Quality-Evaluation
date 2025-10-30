class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        sum_initial = sum(nums1)
        if sum_initial <= x:
            return 0
        
        sum2 = sum(nums2)
        if sum2 == 0:
            # Handle the case where sum2 is zero
            s = sorted(nums1, reverse=True)
            prefix = [0]
            for num in s:
                prefix.append(prefix[-1] + num)
            target = sum_initial - x
            for m in range(1, len(s) + 1):
                if prefix[m] >= target:
                    return m
            return -1
        else:
            # Binary search
            elements = sorted([(nums2[i], nums1[i]) for i in range(len(nums1))], key=lambda x: (-x[0], -x[1]))
            low = 0
            high = 10**18
            ans = -1
            while low <= high:
                mid = (low + high) // 2
                m = min(mid, len(elements))
                sum_reset = 0
                for k in range(m):
                    if k >= len(elements):
                        break
                    s_i = mid - k
                    sum_reset += elements[k][1] + elements[k][0] * s_i
                sum_total = sum_initial + mid * sum2 - sum_reset
                if sum_total <= x:
                    ans = mid
                    high = mid - 1
                else:
                    low = mid + 1
            return ans if ans != -1 else -1