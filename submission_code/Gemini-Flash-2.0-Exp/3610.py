class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        ans = []
        for i in range(n - k + 1):
            sub_array = nums[i:i + k]
            counts = {}
            for num in sub_array:
                counts[num] = counts.get(num, 0) + 1
            
            sorted_counts = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
            
            top_x_elements = []
            for j in range(min(x, len(sorted_counts))):
                top_x_elements.append(sorted_counts[j][0])
            
            x_sum = 0
            for num in sub_array:
                if len(top_x_elements) < x:
                    x_sum += num
                elif num in top_x_elements:
                    x_sum += num
            ans.append(x_sum)
        return ans