from collections import Counter

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        # Count the frequency of each element in the array
        freq = Counter(nums)
        
        # Sort the elements based on their frequency in ascending order
        sorted_freq = sorted(freq.items(), key=lambda x: x[1])
        
        # Initialize the count of distinct elements
        distinct_count = 0
        
        # Iterate through the sorted elements
        for num, count in sorted_freq:
            # If the current element can be modified to become distinct, do so
            if count <= k:
                k -= count
                distinct_count += 1
            # If the current element cannot be modified to become distinct, stop the loop
            else:
                break
        
        # Add the remaining distinct elements to the count
        distinct_count += len(nums) - sum(count for _, count in sorted_freq[:distinct_count])
        
        return distinct_count