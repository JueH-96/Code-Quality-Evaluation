class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        from collections import Counter
        count = Counter(nums)
        nums1, nums2 = set(), set()
        for num in nums:
            if count[num] > 0:
                if count[num] > 1:
                    count[num] -= 2
                    nums1.add(num)
                    nums2.add(num)
                elif num not in nums1:
                    count[num] -= 1
                    nums1.add(num)
                elif num not in nums2:
                    count[num] -= 1
                    nums2.add(num)
                else:
                    return False
        return len(nums1) == len(nums2) == len(nums) // 2