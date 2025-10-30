class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(x for x in nums1 if x != 0)
        sum2 = sum(x for x in nums2 if x != 0)
        count_zeros1 = nums1.count(0)
        count_zeros2 = nums2.count(0)
        
        if count_zeros1 == 0 and count_zeros2 == 0:
            return sum1 if sum1 == sum2 else -1
        
        # Let's assume sum1 <= sum2 without loss of generality
        if sum1 > sum2:
            sum1, sum2 = sum2, sum1
            count_zeros1, count_zeros2 = count_zeros2, count_zeros1
        
        # We need to find the smallest x such that:
        # sum1 + x * count_zeros1 = sum2 + x * count_zeros2
        # x * count_zeros1 - x * count_zeros2 = sum2 - sum1
        # x * (count_zeros1 - count_zeros2) = sum2 - sum1
        # x = (sum2 - sum1) / (count_zeros1 - count_zeros2)
        
        if count_zeros1 == count_zeros2:
            # If the number of zeros is the same, the difference must be zero to be solvable
            if sum1 == sum2:
                return sum1 + count_zeros1  # each zero replaced by at least 1
            else:
                return -1
        else:
            diff = sum2 - sum1
            zero_diff = count_zeros1 - count_zeros2
            if diff % zero_diff == 0:
                x = diff // zero_diff
                if x >= 1:
                    return sum1 + count_zeros1 + sum2 + count_zeros2
                else:
                    return -1
            else:
                return -1