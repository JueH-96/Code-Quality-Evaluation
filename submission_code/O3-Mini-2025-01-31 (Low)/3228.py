from typing import List

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        half = n // 2
        
        # Build sets of distinct numbers for each array.
        set1 = set(nums1)
        set2 = set(nums2)
        
        # Determine elements unique to each array and their common intersection.
        Aonly = set1 - set2
        Bonly = set2 - set1
        common = set1 & set2
        
        # In nums1, we can choose all non-common distinct up to capacity.
        x1 = min(len(Aonly), half)
        r1 = half - x1  # remaining slots in nums1 that must be filled (with common, duplicates add no union)
        
        # Similarly for nums2:
        x2 = min(len(Bonly), half)
        r2 = half - x2
        
        # We can assign up to (r1+r2) distinct common numbers if available
        extra = min(len(common), r1 + r2)
        
        return x1 + x2 + extra

# Example usage:
if __name__ == '__main__':
    sol = Solution()
    print(sol.maximumSetSize([1,2,1,2], [1,1,1,1]))       # Expected output: 2
    print(sol.maximumSetSize([1,2,3,4,5,6], [2,3,2,3,2,3])) # Expected output: 5
    print(sol.maximumSetSize([1,1,2,2,3,3], [4,4,5,5,6,6])) # Expected output: 6