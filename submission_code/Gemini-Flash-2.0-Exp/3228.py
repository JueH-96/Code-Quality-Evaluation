class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        remove = n // 2
        
        s1 = set(nums1)
        s2 = set(nums2)
        
        common = s1.intersection(s2)
        
        only1 = s1.difference(s2)
        only2 = s2.difference(s1)
        
        ans = len(common)
        
        remove1 = min(remove, len(only1))
        ans += remove1
        remove -= remove1
        
        remove2 = min(remove, len(only2))
        ans += remove2
        remove -= remove2
        
        ans += min(len(common), remove)
        
        return ans