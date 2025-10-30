class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        count1 = Counter(nums1)
        count2 = Counter(nums2)
        intersection = 0
        for num in count1:
            if num in count2:
                intersection += min(count1[num], count2[num])
        
        remaining1 = n // 2
        remaining2 = n // 2
        
        set_size = 0
        
        for num in count1:
            if num in count2:
                common = min(count1[num], count2[num])
                removed = min(remaining1, common)
                remaining1 -= removed
                count1[num] -= removed
                
                removed = min(remaining2, common)
                remaining2 -= removed
                count2[num] -= removed

        for num in count1:
            removed = min(remaining1, count1[num])
            remaining1 -= removed
            count1[num] -= removed
            
        for num in count2:
            removed = min(remaining2, count2[num])
            remaining2 -= removed
            count2[num] -= removed
            
        set_size = len(count1) + len(count2)
        
        return set_size