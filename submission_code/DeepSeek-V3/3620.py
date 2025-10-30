from collections import defaultdict

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        # Sort the unique numbers
        sorted_nums = sorted(freq.keys())
        
        # Initialize the result as the number of unique elements
        result = len(sorted_nums)
        
        # Iterate through the sorted numbers and try to make duplicates unique
        for i in range(1, len(sorted_nums)):
            current = sorted_nums[i]
            prev = sorted_nums[i-1]
            # Calculate the difference between current and previous
            diff = current - prev
            if diff <= 2 * k:
                # If the difference is within the range, we can adjust the current number
                # to make it unique
                # The number of duplicates is freq[current] - 1
                duplicates = freq[current] - 1
                # We can adjust at most duplicates numbers
                # Each adjustment can make one duplicate unique
                # So the total number of adjustments is min(duplicates, (diff - 1) // 1)
                # Wait, no. The difference is the gap between current and previous.
                # To make a duplicate unique, we need to shift it by at least 1.
                # So for each duplicate, we can shift it by 1, 2, ..., k.
                # But the total shift must be such that the new number is not equal to any existing number.
                # So for each duplicate, we can shift it by 1, but we need to ensure that the new number is not already in the set.
                # So the maximum number of duplicates we can make unique is min(duplicates, k)
                # Because each shift can be at most k, and we need to shift by at least 1.
                # So the total number of unique elements we can add is min(duplicates, k)
                # So the result increases by min(duplicates, k)
                result += min(duplicates, k)
        
        return result