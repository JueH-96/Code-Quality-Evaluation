class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        from collections import Counter
        N = len(nums)
        N2 = N // 2
        counts = Counter(nums)
        nums1 = set()
        nums2 = set()
        for num in counts:
            if counts[num] >= 2:
                if len(nums1) < N2:
                    nums1.add(num)
                if len(nums2) < N2:
                    nums2.add(num)
        for num in counts:
            if counts[num] >= 1:
                if len(nums1) < N2 and num not in nums1:
                    nums1.add(num)
                elif len(nums2) < N2 and num not in nums2:
                    nums2.add(num)
        return len(nums1) == len(nums2) == N2