from typing import List
class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        from collections import Counter
        n = len(nums)
        answer = []
        for i in range(n - k + 1):
            subarray = nums[i:i + k]
            counts = Counter(subarray)
            # Sort elements by frequency descending, then by value descending
            sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], -item[0]))
            top_x_elements = set([element for element, count in sorted_counts[:x]])
            # Filter the subarray to keep only the top x most frequent elements
            filtered_subarray = [num for num in subarray if num in top_x_elements]
            x_sum = sum(filtered_subarray)
            answer.append(x_sum)
        return answer