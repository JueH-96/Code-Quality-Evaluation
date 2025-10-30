class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def areAlmostEqual(x, y):
            if x == y:
                return True
            
            # Convert to strings to work with digits
            str_x = str(x)
            str_y = str(y)
            
            # Check if we can make x equal to y by swapping digits in x
            for i in range(len(str_x)):
                for j in range(i + 1, len(str_x)):
                    # Swap digits at positions i and j in str_x
                    lst = list(str_x)
                    lst[i], lst[j] = lst[j], lst[i]
                    new_x = int(''.join(lst))
                    if new_x == y:
                        return True
            
            # Check if we can make y equal to x by swapping digits in y
            for i in range(len(str_y)):
                for j in range(i + 1, len(str_y)):
                    # Swap digits at positions i and j in str_y
                    lst = list(str_y)
                    lst[i], lst[j] = lst[j], lst[i]
                    new_y = int(''.join(lst))
                    if new_y == x:
                        return True
            
            return False
        
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if areAlmostEqual(nums[i], nums[j]):
                    count += 1
        
        return count