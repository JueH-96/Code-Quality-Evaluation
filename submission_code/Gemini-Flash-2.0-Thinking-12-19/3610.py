from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        answer = []
        for i in range(n - k + 1):
            sub_array = nums[i:i + k]
            counts = Counter(sub_array)
            sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], -item[0]))
            top_x_elements = [item[0] for item in sorted_counts[:min(x, len(sorted_counts))]]
            current_x_sum = 0
            for val in sub_array:
                if val in top_x_elements:
                    current_x_sum += val
            answer.append(current_x_sum)
        return answer