class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        def calculate_x_sum(subarray):
            # Count frequencies
            freq = {}
            for num in subarray:
                freq[num] = freq.get(num, 0) + 1
            
            # If we have fewer than x distinct elements, return sum of all
            if len(freq) <= x:
                return sum(subarray)
            
            # Sort by frequency (descending), then by value (descending) for ties
            # This gives us the top x most frequent elements
            sorted_items = sorted(freq.items(), key=lambda item: (item[1], item[0]), reverse=True)
            
            # Take top x elements and sum their total occurrences
            total_sum = 0
            for i in range(x):
                num, count = sorted_items[i]
                total_sum += num * count
            
            return total_sum
        
        result = []
        n = len(nums)
        
        # Generate all subarrays of length k
        for i in range(n - k + 1):
            subarray = nums[i:i + k]
            x_sum = calculate_x_sum(subarray)
            result.append(x_sum)
        
        return result