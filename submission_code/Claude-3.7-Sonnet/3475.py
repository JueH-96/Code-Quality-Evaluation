class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        temp = nums.copy()  # Working copy to simulate operations
        operations = 0
        
        # Iterate through the array (except the last 2 elements)
        for i in range(n - 2):
            # If current element is 0, we need to flip it
            if temp[i] == 0:
                operations += 1
                # Flip 3 consecutive elements
                for j in range(i, i+3):
                    temp[j] = 1 - temp[j]
        
        # Check if all elements are 1 after our operations
        if all(x == 1 for x in temp):
            return operations
        else:
            return -1