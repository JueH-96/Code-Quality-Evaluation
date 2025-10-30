class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        def calculate_x_sum(subarray):
            # Count frequencies
            freq = {}
            for num in subarray:
                freq[num] = freq.get(num, 0) + 1
            
            # If we have less than x distinct elements, return sum of all
            if len(freq) <= x:
                return sum(subarray)
            
            # Sort by frequency (descending), then by value (descending)
            # Higher frequency comes first; for ties, higher value comes first
            sorted_items = sorted(freq.items(), key=lambda item: (-item[1], -item[0]))
            
            # Take top x elements and calculate sum
            return sum(num * count for num, count in sorted_items[:x])
        
        return [calculate_x_sum(nums[i:i+k]) for i in range(len(nums) - k + 1)]