class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Check if any element is less than k
        for num in nums:
            if num < k:
                return -1
        
        # Collect all unique elements greater than k
        unique_elements = set()
        for num in nums:
            if num > k:
                unique_elements.add(num)
        
        # Add k to the set to include it in the list
        unique_elements.add(k)
        
        # Sort the elements in decreasing order
        sorted_list = sorted(unique_elements, reverse=True)
        
        # The number of operations is the number of transitions between consecutive elements in the sorted list
        return len(sorted_list) - 1