class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        # Calculate the number of elements to remove from each array
        remove_count = len(nums1) // 2
        
        # Create frequency dictionaries for both arrays
        freq1 = {}
        freq2 = {}
        
        for num in nums1:
            if num in freq1:
                freq1[num] += 1
            else:
                freq1[num] = 1
        
        for num in nums2:
            if num in freq2:
                freq2[num] += 1
            else:
                freq2[num] = 1
        
        # Sort the elements by frequency in descending order
        sorted_nums1 = sorted(freq1.items(), key=lambda x: -x[1])
        sorted_nums2 = sorted(freq2.items(), key=lambda x: -x[1])
        
        # Remove elements from nums1
        remaining_nums1 = []
        for num, freq in sorted_nums1:
            if remove_count > 0:
                if freq <= remove_count:
                    remove_count -= freq
                else:
                    remaining_nums1.append(num)
                    remove_count = 0
            else:
                remaining_nums1.append(num)
        
        # Reset remove_count for nums2
        remove_count = len(nums2) // 2
        
        # Remove elements from nums2
        remaining_nums2 = []
        for num, freq in sorted_nums2:
            if remove_count > 0:
                if freq <= remove_count:
                    remove_count -= freq
                else:
                    remaining_nums2.append(num)
                    remove_count = 0
            else:
                remaining_nums2.append(num)
        
        # Create a set from the remaining elements
        result_set = set(remaining_nums1 + remaining_nums2)
        
        # Return the size of the set
        return len(result_set)