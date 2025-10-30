class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        from collections import Counter
        
        n = len(nums1)
        k = n // 2
        
        # Count frequency of each number in both arrays
        count1 = Counter(nums1)
        count2 = Counter(nums2)
        
        # Create a list of tuples (frequency in nums1, frequency in nums2, value)
        combined_counts = []
        all_values = set(nums1 + nums2)
        for value in all_values:
            combined_counts.append((count1.get(value, 0), count2.get(value, 0), value))
        
        # Sort by the sum of frequencies in descending order
        combined_counts.sort(key=lambda x: -(x[0] + x[1]))
        
        # We need to maximize the number of unique elements in the final set
        # Start by trying to include as many high-frequency elements as possible
        remaining_k1 = k
        remaining_k2 = k
        result_set = set()
        
        for c1, c2, value in combined_counts:
            if remaining_k1 == 0 and remaining_k2 == 0:
                break
            
            # Determine how many of this value we can take from each array
            take_from_1 = min(c1, remaining_k1)
            take_from_2 = min(c2, remaining_k2)
            
            # Add to result set if we can take any from either array
            if take_from_1 > 0 or take_from_2 > 0:
                result_set.add(value)
            
            # Update remaining capacity
            remaining_k1 -= take_from_1
            remaining_k2 -= take_from_2
        
        return len(result_set)