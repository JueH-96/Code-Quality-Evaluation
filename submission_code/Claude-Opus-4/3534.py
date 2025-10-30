class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def canBecomeEqual(x, y):
            # Convert to strings to work with digits
            str_x = str(x)
            str_y = str(y)
            
            # If already equal
            if str_x == str_y:
                return True
            
            # Try swapping digits in x to match y
            for i in range(len(str_x)):
                for j in range(i + 1, len(str_x)):
                    # Swap digits at positions i and j
                    chars = list(str_x)
                    chars[i], chars[j] = chars[j], chars[i]
                    if ''.join(chars) == str_y:
                        return True
            
            # Try swapping digits in y to match x
            for i in range(len(str_y)):
                for j in range(i + 1, len(str_y)):
                    # Swap digits at positions i and j
                    chars = list(str_y)
                    chars[i], chars[j] = chars[j], chars[i]
                    if ''.join(chars) == str_x:
                        return True
            
            return False
        
        count = 0
        n = len(nums)
        
        # Check all pairs (i, j) where i < j
        for i in range(n):
            for j in range(i + 1, n):
                if canBecomeEqual(nums[i], nums[j]):
                    count += 1
        
        return count