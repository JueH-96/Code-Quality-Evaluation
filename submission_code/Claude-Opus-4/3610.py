class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        from collections import Counter
        
        result = []
        n = len(nums)
        
        for i in range(n - k + 1):
            # Get the subarray
            subarray = nums[i:i + k]
            
            # Count occurrences
            counter = Counter(subarray)
            
            # Sort by frequency (descending), then by value (descending) for tiebreaker
            # We want: higher frequency first, then higher value for same frequency
            sorted_items = sorted(counter.items(), key=lambda item: (-item[1], -item[0]))
            
            # Take top x elements
            top_x = sorted_items[:x]
            
            # Calculate x-sum: sum of (element * its count) for top x elements
            x_sum = sum(element * count for element, count in top_x)
            
            result.append(x_sum)
        
        return result