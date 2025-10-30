class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
        count = 0
        # Loop over all possible pairs (i, j)
        for num1 in nums1:
            for num2 in nums2:
                if num2 * k != 0 and num1 % (num2 * k) == 0:
                    count += 1
        return count