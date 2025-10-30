class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def can_be_equal(num1, num2):
            if num1 == num2:
                return True
                
            # Convert numbers to strings for digit manipulation
            s1 = str(num1)
            s2 = str(num2)
            
            # If lengths are different, they can't be equal
            if len(s1) != len(s2):
                return False
                
            # Try swapping each pair of digits in num1
            for i in range(len(s1)):
                for j in range(i+1, len(s1)):
                    # Create new number by swapping digits at i and j
                    s1_list = list(s1)
                    s1_list[i], s1_list[j] = s1_list[j], s1_list[i]
                    if ''.join(s1_list) == s2:
                        return True
                        
            # Try swapping each pair of digits in num2
            for i in range(len(s2)):
                for j in range(i+1, len(s2)):
                    # Create new number by swapping digits at i and j
                    s2_list = list(s2)
                    s2_list[i], s2_list[j] = s2_list[j], s2_list[i]
                    if ''.join(s2_list) == s1:
                        return True
                        
            return False
            
        count = 0
        n = len(nums)
        
        # Check each pair of numbers
        for i in range(n):
            for j in range(i+1, n):
                if can_be_equal(nums[i], nums[j]):
                    count += 1
                    
        return count