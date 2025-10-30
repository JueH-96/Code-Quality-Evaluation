class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        k = n // 2
        
        # Get unique elements from both arrays
        set1 = set(nums1)
        set2 = set(nums2)
        
        # Common elements between the two sets
        common = set1 & set2
        
        # Unique elements in each set
        unique1 = set1 - common
        unique2 = set2 - common
        
        # Calculate the maximum number of unique elements we can keep
        # We can keep min(k, len(unique1)) from unique1 and min(k, len(unique2)) from unique2
        # For common elements, we can keep min(2*k - (kept_unique1 + kept_unique2), len(common))
        kept_unique1 = min(k, len(unique1))
        kept_unique2 = min(k, len(unique2))
        
        # Remaining slots for common elements
        remaining = 2 * k - (kept_unique1 + kept_unique2)
        kept_common = min(remaining, len(common))
        
        # Total unique elements in the set
        total_unique = kept_unique1 + kept_unique2 + kept_common
        
        return total_unique