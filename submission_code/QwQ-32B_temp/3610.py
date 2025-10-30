from typing import List

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        answer = []
        for i in range(n - k + 1):
            sub = nums[i:i+k]
            # Compute frequency of each element in the subarray
            freq = {}
            for num in sub:
                if num in freq:
                    freq[num] += 1
                else:
                    freq[num] = 1
            # Create list of (element, count) pairs
            items = list(freq.items())
            # Sort by descending count, then descending element value
            items.sort(key=lambda item: (-item[1], -item[0]))
            # Get the top x elements (or all if fewer than x)
            top_elements = [item[0] for item in items[:x]]
            top_set = set(top_elements)
            # Calculate the sum of elements in the subarray that are in top_elements
            current_sum = 0
            for num in sub:
                if num in top_set:
                    current_sum += num
            answer.append(current_sum)
        return answer