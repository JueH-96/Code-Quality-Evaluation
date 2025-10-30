class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        from collections import Counter
        
        # Count the frequency of each element in nums
        freq = Counter(nums)
        
        # Create a list of elements sorted by their frequency in descending order
        elements_by_freq = sorted(freq.items(), key=lambda x: -x[1])
        
        # Initialize a set to keep track of used elements
        used_elements = set()
        
        # Iterate over the elements sorted by frequency
        for num, count in elements_by_freq:
            # Try to add the element itself and its possible variations within the range [-k, k]
            for i in range(-k, k + 1):
                new_num = num + i
                if new_num not in used_elements:
                    used_elements.add(new_num)
                    break
        
        # The size of the used_elements set is the maximum number of distinct elements
        return len(used_elements)