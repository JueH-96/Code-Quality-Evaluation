from collections import Counter

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        counter1 = Counter(nums1)
        counter2 = Counter(nums2)
        common_elements = counter1 & counter2
        
        n = len(nums1)
        
        # Calculate the total possible common elements count after removals
        common_count = sum(common_elements.values()) * 2
        
        # Calculate the maximum possible size of the set s
        set_size = min(2 * n - common_count, 2 * n // 2)
        
        # If there are no common elements, the set size could be the full length of either array
        return set_size if common_count else n