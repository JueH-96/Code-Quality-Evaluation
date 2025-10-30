from collections import Counter

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        unique_nums1 = set(nums1)
        unique_nums2 = set(nums2)
        u1_only = unique_nums1 - unique_nums2
        u2_only = unique_nums2 - unique_nums1
        u_both = unique_nums1.intersection(unique_nums2)
        
        counts1 = Counter(nums1)
        counts2 = Counter(nums2)
        
        kept_counts1 = Counter()
        kept_counts2 = Counter()
        remaining_slots1 = n // 2
        remaining_slots2 = n // 2
        result_set = set()
        
        for x in u1_only:
            if remaining_slots1 > 0 and counts1[x] > 0:
                kept_counts1[x] = 1
                remaining_slots1 -= 1
                result_set.add(x)
                
        for x in u2_only:
            if remaining_slots2 > 0 and counts2[x] > 0:
                kept_counts2[x] = 1
                remaining_slots2 -= 1
                result_set.add(x)
                
        for x in u_both:
            if remaining_slots1 > 0 and counts1[x] > 0:
                kept_counts1[x] = 1
                remaining_slots1 -= 1
                result_set.add(x)
            if remaining_slots2 > 0 and counts2[x] > 0:
                kept_counts2[x] = 1
                remaining_slots2 -= 1
                result_set.add(x)
                
        return len(result_set)