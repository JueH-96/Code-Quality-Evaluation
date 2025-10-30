class Solution:
    def countPairs(self, nums: List[int]) -> int:
        count = 0
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if self.areAlmostEqual(nums[i], nums[j]):
                    count += 1
        return count
    
    def areAlmostEqual(self, x: int, y: int) -> bool:
        x_digits = list(str(x))
        y_digits = list(str(y))
        if x_digits == y_digits:
            return True
        
        diff_indices = []
        for i in range(len(x_digits)):
            if x_digits[i] != y_digits[i]:
                diff_indices.append(i)
        
        if len(diff_indices) == 2:
            x_digits[diff_indices[0]], x_digits[diff_indices[1]] = x_digits[diff_indices[1]], x_digits[diff_indices[0]]
            return ''.join(x_digits) == ''.join(y_digits)
        
        return False