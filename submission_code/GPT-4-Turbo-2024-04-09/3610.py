class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        from collections import Counter
        
        def calculate_x_sum(subarray: List[int]) -> int:
            count = Counter(subarray)
            # Sort by frequency, then by value if frequencies are the same
            sorted_items = sorted(count.items(), key=lambda item: (-item[1], item[0]))
            # Keep only the top x elements
            top_x_elements = sorted_items[:x]
            # Calculate the sum of the resulting array
            result_sum = 0
            for value, freq in top_x_elements:
                result_sum += value * freq
            return result_sum
        
        n = len(nums)
        result = []
        # Iterate over each subarray of length k
        for i in range(n - k + 1):
            subarray = nums[i:i + k]
            result.append(calculate_x_sum(subarray))
        
        return result