from collections import defaultdict

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        answer = []
        for i in range(n - k + 1):
            subarray = nums[i:i+k]
            freq = defaultdict(int)
            for num in subarray:
                freq[num] += 1
            # Sort by frequency descending, then by value descending
            sorted_items = sorted(freq.items(), key=lambda item: (-item[1], -item[0]))
            # Keep top x elements
            top_x = sorted_items[:x]
            # Sum all elements in subarray that are in top_x
            sum_x = 0
            for num in subarray:
                if num in [item[0] for item in top_x]:
                    sum_x += num
            answer.append(sum_x)
        return answer