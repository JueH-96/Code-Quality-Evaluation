class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        ans = []
        for i in range(n - k + 1):
            sub_array = nums[i:i + k]
            counts = {}
            for num in sub_array:
                counts[num] = counts.get(num, 0) + 1
            
            sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], -item[0]))
            
            x_sum = 0
            
            if len(counts) <= x:
                x_sum = sum(sub_array)
            else:
                top_x = sorted_counts[:x]
                top_x_elements = set([item[0] for item in top_x])
                for num in sub_array:
                    if num in top_x_elements:
                        x_sum += num
            ans.append(x_sum)
        return ans