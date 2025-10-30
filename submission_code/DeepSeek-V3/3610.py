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
            sorted_freq = sorted(freq.items(), key=lambda item: (-item[1], -item[0]))
            # Select top x elements
            selected = sorted_freq[:x]
            # Calculate the sum
            total = 0
            for num, cnt in selected:
                total += num * cnt
            answer.append(total)
        return answer