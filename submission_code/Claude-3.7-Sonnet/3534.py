class Solution:
    def countPairs(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                if self.are_almost_equal(nums[i], nums[j]):
                    count += 1
        
        return count
    
    def are_almost_equal(self, x, y):
        # Check if x and y are already equal
        if x == y:
            return True
        
        x_str = str(x)
        y_str = str(y)
        
        # Check if swapping any two digits in x can make it equal to y
        for i in range(len(x_str)):
            for j in range(i+1, len(x_str)):
                # Swap digits at positions i and j in x_str
                x_swapped = list(x_str)
                x_swapped[i], x_swapped[j] = x_swapped[j], x_swapped[i]
                x_swapped_str = ''.join(x_swapped)
                
                # Convert to integer to handle leading zeros
                x_swapped_int = int(x_swapped_str)
                
                if x_swapped_int == y:
                    return True
        
        # Check if swapping any two digits in y can make it equal to x
        for i in range(len(y_str)):
            for j in range(i+1, len(y_str)):
                # Swap digits at positions i and j in y_str
                y_swapped = list(y_str)
                y_swapped[i], y_swapped[j] = y_swapped[j], y_swapped[i]
                y_swapped_str = ''.join(y_swapped)
                
                # Convert to integer to handle leading zeros
                y_swapped_int = int(y_swapped_str)
                
                if y_swapped_int == x:
                    return True
        
        return False