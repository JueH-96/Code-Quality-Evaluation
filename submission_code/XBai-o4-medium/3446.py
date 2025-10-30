class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
        count = 0
        for num in nums1:
            for num2 in nums2:
                product = num2 * k
                if num % product == 0:
                    count += 1
        return count