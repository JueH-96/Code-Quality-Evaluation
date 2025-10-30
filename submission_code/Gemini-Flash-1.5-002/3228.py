class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        count1 = Counter(nums1)
        count2 = Counter(nums2)
        common = 0
        uncommon1 = 0
        uncommon2 = 0
        for num in set(nums1 + nums2):
            c1 = count1[num]
            c2 = count2[num]
            common += min(c1, c2)
            uncommon1 += max(0, c1 - c2)
            uncommon2 += max(0, c2 - c1)
        n = len(nums1)
        remove = (uncommon1 + uncommon2) // 2
        return common + min(common, remove)