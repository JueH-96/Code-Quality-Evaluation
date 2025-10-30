from collections import Counter

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        k = n // 2
        
        # Count the frequency of each element in both arrays
        count1 = Counter(nums1)
        count2 = Counter(nums2)
        
        # Calculate the unique elements in both arrays
        unique1 = set(nums1)
        unique2 = set(nums2)
        
        # Calculate the common elements
        common = unique1 & unique2
        
        # Calculate the number of unique elements in each array that are not in the other
        only1 = unique1 - unique2
        only2 = unique2 - unique1
        
        # Calculate the maximum number of unique elements we can keep from each array
        # We can keep at most k elements from each array
        # For common elements, we need to decide how to split them between the two arrays
        
        # First, count how many common elements we can keep
        # We can keep at most min(count1[c], count2[c]) for each common element
        # But since we need to remove k elements from each array, we need to manage the counts
        
        # The total number of common elements we can keep is limited by the sum of the minimum counts
        # and the fact that we can only keep k elements from each array
        
        # Let's calculate the maximum number of common elements we can keep
        # We can keep at most min(sum(min(count1[c], count2[c]) for c in common), 2 * k)
        
        # But to maximize the unique set, we should prioritize keeping unique elements first
        
        # So, first, we keep all unique elements from only1 and only2
        # Then, we try to keep as many common elements as possible
        
        # Calculate the number of unique elements we can keep from only1 and only2
        # We can keep at most k elements from each array
        # So, the number of unique elements we can keep from only1 is min(len(only1), k)
        # Similarly for only2
        
        keep_only1 = min(len(only1), k)
        keep_only2 = min(len(only2), k)
        
        # The remaining slots for each array are k - keep_only1 and k - keep_only2
        # We can use these slots to keep common elements
        
        # The total number of common elements we can keep is limited by the sum of the remaining slots
        # and the sum of the minimum counts of the common elements
        
        # Calculate the sum of the minimum counts of the common elements
        sum_min_counts = 0
        for c in common:
            sum_min_counts += min(count1[c], count2[c])
        
        # The total number of common elements we can keep is min(sum_min_counts, (k - keep_only1) + (k - keep_only2))
        keep_common = min(sum_min_counts, (k - keep_only1) + (k - keep_only2))
        
        # The total number of unique elements in the set is keep_only1 + keep_only2 + keep_common
        total_unique = keep_only1 + keep_only2 + keep_common
        
        return total_unique