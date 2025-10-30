class Solution:
    def minCost(self, nums: List[int]) -> int:
        total_cost = 0
        
        while nums:
            if len(nums) >= 3:
                # Take the first three elements
                first_three = nums[:3]
                # Sort to find the two largest
                first_three.sort()
                # Remove the two largest elements
                total_cost += first_three[1]  # second largest
                nums.remove(first_three[1])
                nums.remove(first_three[2])
            else:
                # If fewer than 3 elements remain, remove all
                total_cost += max(nums)
                nums.clear()  # Clear the list as all elements are removed
        
        return total_cost