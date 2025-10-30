class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Convert all numbers to strings
        str_nums = [str(num) for num in nums]
        num_digits = len(str_nums[0])
        
        total_sum = 0
        
        # For each digit position
        for pos in range(num_digits):
            # Count frequency of each digit at this position
            count = [0] * 10
            for s in str_nums:
                digit = int(s[pos])
                count[digit] += 1
            
            # Calculate number of pairs with different digits at this position
            sum_of_squares = sum(c * c for c in count)
            pairs_with_different = (n * n - sum_of_squares) // 2
            
            total_sum += pairs_with_different
        
        return total_sum