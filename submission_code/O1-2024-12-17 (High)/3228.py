class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        # n is even and both arrays have the same length
        n = len(nums1)
        half = n // 2
        
        # Convert nums1 and nums2 to sets to identify unique elements
        set1 = set(nums1)
        set2 = set(nums2)
        
        # A: elements unique to nums1
        # B: elements unique to nums2
        # C: elements common to both
        A = set1 - set2
        B = set2 - set1
        C = set1 & set2
        
        # We can take at most half elements from nums1 and half from nums2.
        # Elements in A must come from nums1; elements in B must come from nums2.
        keepA = min(len(A), half)  # how many distinct A-elements we keep in nums1
        keepB = min(len(B), half)  # how many distinct B-elements we keep in nums2
        
        # After picking those, the leftover capacity is how many distinct C-elements we can still place
        # across both arrays (in sum).
        leftover = (half - keepA) + (half - keepB)
        
        # We can only keep as many distinct C-elements as we have leftover capacity.
        keepC = min(len(C), leftover)
        
        return keepA + keepB + keepC