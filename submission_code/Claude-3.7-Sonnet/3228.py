class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        half_n = n // 2
        
        # Convert to sets to get unique elements
        set1 = set(nums1)
        set2 = set(nums2)
        
        # We can keep at most half_n unique elements from each array
        count1 = min(len(set1), half_n)
        count2 = min(len(set2), half_n)
        
        # Total number of unique elements after combining
        total_unique = len(set1.union(set2))
        
        # We can't have more than count1 + count2 elements in the final set
        return min(total_unique, count1 + count2)