class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        result = []
        
        for i in range(n - k + 1):
            # Count occurrences in the window
            freq = {}
            for j in range(i, i + k):
                freq[nums[j]] = freq.get(nums[j], 0) + 1
            
            # Sort by frequency (descending) and value (descending)
            sorted_items = sorted(freq.items(), key=lambda item: (-item[1], -item[0]))
            
            # Take top x elements
            top_x = sorted_items[:x]
            
            # Calculate x-sum
            x_sum = 0
            for value, count in top_x:
                x_sum += value * count
            
            result.append(x_sum)
        
        return result