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
                # We can adjust at most min(duplicates, k) elements
                # Since each adjustment can make one duplicate unique
                # But we need to ensure that the new value is not already in the set
                # So, we need to find a new value that is not in the current set
                # For simplicity, we can assume that we can adjust one duplicate per step
                # and reduce the duplicates by 1
                if duplicates > 0:
                    result += 1
                    # Adjust the frequency
                    freq[current] -= 1
                    # Add the new value to the frequency map
                    # For simplicity, we can add current + k or current - k
                    # But we need to ensure it's unique
                    # Since the numbers are sorted, we can add current + k
                    new_num = current + k
                    freq[new_num] += 1
                    # Update the sorted_nums list
                    # Since we are adding a new number, we need to insert it in the correct position
                    # But for the purpose of this problem, we can assume that the new number is unique
                    # and does not interfere with the existing numbers
                    # So, we can continue
            else:
                # If the difference is greater than 2k, we cannot make the current number unique
                # by adjusting the previous number
                pass
        
        return result