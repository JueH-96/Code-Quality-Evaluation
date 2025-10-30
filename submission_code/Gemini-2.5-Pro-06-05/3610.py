import collections
from typing import List

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        answer = []
        
        # Iterate through all subarrays of length k
        for i in range(n - k + 1):
            subarray = nums[i : i + k]
            
            # --- Start of x-sum calculation for the subarray ---
            
            # 1. Count occurrences of all elements in the subarray.
            counts = collections.Counter(subarray)
            
            # Note: if an array has less than x distinct elements, its x-sum is the sum of the array.
            if len(counts) < x:
                answer.append(sum(subarray))
                continue
            
            # 2. Keep only the occurrences of the top x most frequent elements.
            
            # Convert the counter to a list of (element, count) pairs.
            freq_list = list(counts.items())
            
            # Sort the list to determine frequency ranking.
            # Primary sort key: count (descending).
            # Secondary sort key (tie-breaker): element value (descending).
            # We use a lambda function with a tuple (item[1], item[0]) which corresponds to (count, element)
            # and sort in reverse to achieve the desired order.
            freq_list.sort(key=lambda item: (item[1], item[0]), reverse=True)
            
            # Get the top x items based on the sorted list.
            top_x_items = freq_list[:x]
            
            # 3. Calculate the sum of the resulting array.
            # This is equivalent to summing (element * count) for the top x elements.
            current_x_sum = 0
            for element, count in top_x_items:
                current_x_sum += element * count
            
            answer.append(current_x_sum)
            
        return answer