class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def canBeEqual(x, y):
            # If already equal
            if x == y:
                return True
            
            # Convert to strings for digit manipulation
            str_x = str(x)
            str_y = str(y)
            
            # Try swapping digits in x
            for i in range(len(str_x)):
                for j in range(i + 1, len(str_x)):
                    # Swap digits i and j in str_x
                    chars = list(str_x)
                    chars[i], chars[j] = chars[j], chars[i]
                    new_num = int(''.join(chars))
                    if new_num == y:
                        return True
            
            # Try swapping digits in y
            for i in range(len(str_y)):
                for j in range(i + 1, len(str_y)):
                    # Swap digits i and j in str_y
                    chars = list(str_y)
                    chars[i], chars[j] = chars[j], chars[i]
                    new_num = int(''.join(chars))
                    if new_num == x:
                        return True
            
            return False
        
        count = 0
        n = len(nums)
        
        for i in range(n):
            for j in range(i + 1, n):
                if canBeEqual(nums[i], nums[j]):
                    count += 1
        
        return count