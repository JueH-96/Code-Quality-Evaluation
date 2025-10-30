class Solution:
    def maxScore(self, nums: list[int], x: int) -> int:
        # Initialize the maximum score with the first element
        max_score = nums[0]
        current_score = nums[0]
        
        # Initialize variables to track the last visited index and its parity
        last_visited_index = 0
        last_visited_parity = nums[0] % 2
        
        # Iterate through the array starting from the second element
        for i in range(1, len(nums)):
            # Check if the parity of the current element is different from the last visited
            if nums[i] % 2 != last_visited_parity:
                # Subtract the penalty from the current score
                current_score -= x
            # Add the current element's value to the current score
            current_score += nums[i]
            # Update the maximum score if the current score is greater
            max_score = max(max_score, current_score)
            # Update the last visited index and its parity
            last_visited_index = i
            last_visited_parity = nums[i] % 2
        
        return max_score

# Example usage:
# sol = Solution()
# print(sol.maxScore([2,3,6,1,9,2], 5)) # Output: 13
# print(sol.maxScore([2,4,6,8], 3)) # Output: 20