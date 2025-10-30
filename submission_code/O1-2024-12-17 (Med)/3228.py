class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        # n is even and the two arrays have the same length
        n = len(nums1)
        
        # Get distinct sets and their sizes
        set1 = set(nums1)
        set2 = set(nums2)
        a = len(set1)  # number of distinct elements in nums1
        b = len(set2)  # number of distinct elements in nums2
        c = len(set1.intersection(set2))  # number of elements common to both sets
        
        # We must keep exactly n//2 elements from nums1 and n//2 from nums2
        # but we aim to maximize the number of distinct elements overall.
        #
        # Let x1 = min(n//2, a) = maximum distinct we can keep from nums1
        #     x2 = min(n//2, b) = maximum distinct we can keep from nums2
        #
        # If the arrays share c distinct elements, some of these intersections
        # might be "forced" (i.e., we may have to pick them in both arrays if
        # one array does not have enough distinct elements outside the intersection).
        #
        # forced1 = max(0, x1 - (a - c)) = how many intersection elements we need
        #           to pick from array1 because we don't have enough "non-intersecting"
        #           distinct elements to fill x1.
        # forced2 = max(0, x2 - (b - c)) = likewise for array2.
        #
        # Together we must pick at least forced1 + forced2 intersection elements.
        # However, we only have c distinct intersection elements total, so the
        # actual overlap in distinct picks is min(c, forced1 + forced2).
        #
        # Hence, the maximum distinct union is:
        #   union_size = x1 + x2 - min(c, forced1 + forced2)
        #
        # Finally, we cannot exceed n total distinct elements, so we take:
        #   answer = min(n, union_size)
        
        x1 = min(n // 2, a)
        x2 = min(n // 2, b)
        
        forced1 = max(0, x1 - (a - c))
        forced2 = max(0, x2 - (b - c))
        
        overlap_forced = forced1 + forced2
        union_size = x1 + x2 - min(c, overlap_forced)
        
        return min(n, union_size)