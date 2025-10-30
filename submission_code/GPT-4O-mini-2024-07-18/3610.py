from collections import Counter
from typing import List

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        answer = []
        
        for i in range(n - k + 1):
            subarray = nums[i:i + k]
            count = Counter(subarray)
            # Get the most common elements, sorted by frequency and then by value
            most_common = count.most_common()
            most_common.sort(key=lambda item: (-item[1], -item[0]))  # Sort by frequency desc, value desc
            
            # Keep only the top x elements
            top_x_elements = most_common[:x]
            # Calculate the x-sum
            x_sum = sum(value * count for value, count in top_x_elements)
            answer.append(x_sum)
        
        return answer