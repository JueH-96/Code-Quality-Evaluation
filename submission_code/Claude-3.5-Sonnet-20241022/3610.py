class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        def get_x_sum(arr):
            # Count frequencies
            freq = {}
            for num in arr:
                freq[num] = freq.get(num, 0) + 1
            
            # Sort by frequency and value
            # For same frequency, bigger value comes first
            sorted_items = sorted(freq.items(), key=lambda x: (-x[1], -x[0]))
            
            # Keep only top x most frequent elements
            x_sum = 0
            used_elements = set()
            
            # If less than x distinct elements, return sum of array
            if len(freq) < x:
                return sum(arr)
            
            # Add contribution of top x elements
            for i in range(x):
                num, count = sorted_items[i]
                used_elements.add(num)
                x_sum += num * count
            
            return x_sum
        
        # Initialize result array
        result = []
        
        # Calculate x-sum for each window of size k
        for i in range(len(nums) - k + 1):
            window = nums[i:i + k]
            x_sum = get_x_sum(window)
            result.append(x_sum)
        
        return result