class Solution:
    def maxSum(self, nums: List[int]) -> int:
        n = len(nums)
        max_sum = float('-inf')
        
        # Try all possible starting positions
        for i in range(n):
            seen = set()
            unique_elements = []
            
            # Extend the range and collect unique elements
            for j in range(i, n):
                if nums[j] not in seen:
                    seen.add(nums[j])
                    unique_elements.append(nums[j])
                
                # Find max sum contiguous subarray in unique_elements
                # This represents selecting a subarray after deletions
                for start in range(len(unique_elements)):
                    current_sum = 0
                    for end in range(start, len(unique_elements)):
                        current_sum += unique_elements[end]
                        max_sum = max(max_sum, current_sum)
        
        return max_sum