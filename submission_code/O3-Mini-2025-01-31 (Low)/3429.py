class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        # If the list has one element, it's always special.
        if len(nums) < 2:
            return True
        
        # Iterate through adjacent pairs and check for different parities
        for i in range(len(nums) - 1):
            # Check parity: even if number % 2 == 0, odd otherwise
            if (nums[i] % 2) == (nums[i + 1] % 2):
                return False
        return True