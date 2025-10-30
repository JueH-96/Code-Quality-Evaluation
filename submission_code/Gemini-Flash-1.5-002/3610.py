class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        answer = []
        for i in range(n - k + 1):
            sub_array = nums[i:i + k]
            counts = {}
            for num in sub_array:
                counts[num] = counts.get(num, 0) + 1
            
            sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], -item[0]))
            
            top_x = sorted_counts[:min(x, len(sorted_counts))]
            
            x_sum = 0
            for num, count in top_x:
                x_sum += num * count
            answer.append(x_sum)
        return answer