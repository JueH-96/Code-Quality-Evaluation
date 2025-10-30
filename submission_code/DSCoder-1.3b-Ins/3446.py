class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
        count = collections.Counter(nums1[i] + nums2[j]*k for i in range(len(nums1)) for j in range(len(nums2)))
        return sum(v*(v-1)//2 for v in count.values())