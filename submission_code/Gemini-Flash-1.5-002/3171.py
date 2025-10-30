class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        zeros1 = nums1.count(0)
        zeros2 = nums2.count(0)

        if zeros1 == 0 and zeros2 == 0 and sum1 != sum2:
            return -1

        left = 1
        right = 10**7  # Adjust upper bound if needed based on constraints

        while left <= right:
            mid = (left + right) // 2
            total_sum = sum1 + sum2 + zeros1 + zeros2
            if zeros1 > zeros2:
                if (total_sum + zeros1 - zeros2) % 2 != 0:
                    left = mid + 1
                    continue
                target_sum = (total_sum + zeros1 - zeros2) // 2
                if target_sum >= sum1 + zeros1 * mid and target_sum >= sum2:
                    right = mid -1
                else:
                    left = mid + 1
            elif zeros2 > zeros1:
                if (total_sum + zeros2 - zeros1) % 2 != 0:
                    left = mid + 1
                    continue
                target_sum = (total_sum + zeros2 - zeros1) // 2
                if target_sum >= sum2 + zeros2 * mid and target_sum >= sum1:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if (sum1 + sum2) % 2 != 0:
                    return -1
                if sum1 + sum2 >= sum1 + zeros1 * mid and sum1 + sum2 >= sum2 + zeros2 * mid:
                    return sum1 + sum2
                else:
                    return -1

        if left > 10**7:
            return -1
        
        total_sum = sum1 + sum2 + zeros1 + zeros2
        if zeros1 > zeros2:
            target_sum = (total_sum + zeros1 - zeros2) // 2
            if (total_sum + zeros1 - zeros2) % 2 != 0:
                return -1
            if target_sum >= sum1 + zeros1 * left and target_sum >= sum2:
                return target_sum * 2
            else:
                return -1
        elif zeros2 > zeros1:
            target_sum = (total_sum + zeros2 - zeros1) // 2
            if (total_sum + zeros2 - zeros1) % 2 != 0:
                return -1
            if target_sum >= sum2 + zeros2 * left and target_sum >= sum1:
                return target_sum * 2
            else:
                return -1
        else:
            if (sum1 + sum2) % 2 != 0:
                return -1
            return sum1 + sum2