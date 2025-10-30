class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        nums1_set = set(nums1)
        nums2_set = set(nums2)
        
        # Find the elements that are present in both nums1 and nums2
        common_elements = nums1_set.intersection(nums2_set)
        
        # Count the frequency of each common element
        freq = {}
        for num in common_elements:
            freq[num] = min(nums1.count(num), nums2.count(num))
        
        # Sort the common elements by their frequency in descending order
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        # Remove the n/2 least frequent elements from both nums1 and nums2
        for i in range(n // 2):
            num, count = sorted_freq[i]
            nums1.remove(num)
            nums2.remove(num)
        
        # Create the set s with the remaining elements
        s = set(nums1 + nums2)
        
        return len(s)