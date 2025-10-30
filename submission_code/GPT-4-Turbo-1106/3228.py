from typing import List
from collections import Counter

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        # Count the frequency of each number in both arrays
        counter1 = Counter(nums1)
        counter2 = Counter(nums2)
        
        # Combine the counts and sort by the total frequency in descending order
        combined_counts = [(num, counter1[num] + counter2[num]) for num in set(nums1 + nums2)]
        combined_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate the number of elements to be removed from each array
        remove_count = len(nums1) // 2
        
        # Initialize the set s
        s = set()
        
        # Iterate over the combined counts
        for num, count in combined_counts:
            # If we still need to remove elements from both arrays
            if remove_count > 0:
                # Calculate the number of occurrences to remove from each array
                remove_from_nums1 = min(remove_count, counter1[num])
                remove_from_nums2 = min(remove_count, counter2[num])
                
                # Update the remove count
                remove_count -= max(remove_from_nums1, remove_from_nums2)
                
                # Update the counters
                counter1[num] -= remove_from_nums1
                counter2[num] -= remove_from_nums2
            # If the number still exists in either array, add it to the set
            if counter1[num] > 0 or counter2[num] > 0:
                s.add(num)
        
        # Return the size of the set
        return len(s)