class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        from collections import Counter
        
        # Calculate the number of elements to remove
        n = len(nums1)
        remove_count = n // 2
        
        # Count the frequency of each element in both arrays
        count1 = Counter(nums1)
        count2 = Counter(nums2)
        
        # Create a combined counter to get the total frequency of each element
        combined_count = count1 + count2
        
        # Sort elements by their combined frequency in descending order
        sorted_elements = sorted(combined_count.keys(), key=lambda x: combined_count[x], reverse=True)
        
        # Initialize the set to store the remaining elements
        remaining_elements = set()
        
        # Iterate through the sorted elements and add them to the set
        for element in sorted_elements:
            if count1[element] > 0 and remove_count > 0:
                remove_from_nums1 = min(count1[element], remove_count)
                count1[element] -= remove_from_nums1
                remove_count -= remove_from_nums1
            
            if count2[element] > 0 and remove_count > 0:
                remove_from_nums2 = min(count2[element], remove_count)
                count2[element] -= remove_from_nums2
                remove_count -= remove_from_nums2
            
            if count1[element] > 0 or count2[element] > 0:
                remaining_elements.add(element)
        
        return len(remaining_elements)