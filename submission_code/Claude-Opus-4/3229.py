class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(n):
            s = str(n)
            return s == s[::-1]
        
        def generate_palindromes_near(target, count=5):
            """Generate palindromic numbers near the target value"""
            palindromes = []
            
            # Check numbers around the target
            left = target
            right = target
            
            while len(palindromes) < count * 2:
                if left >= 0 and is_palindrome(left):
                    palindromes.append(left)
                if right != left and right < 10**9 and is_palindrome(right):
                    palindromes.append(right)
                left -= 1
                right += 1
            
            return palindromes
        
        def calculate_cost(nums, target):
            return sum(abs(num - target) for num in nums)
        
        # Sort the array to find median
        sorted_nums = sorted(nums)
        n = len(nums)
        
        # Find median
        if n % 2 == 1:
            median = sorted_nums[n // 2]
        else:
            median = (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) // 2
        
        # Generate palindromic numbers near the median
        palindromes = generate_palindromes_near(median, 100)
        
        # Also check some edge cases
        palindromes.extend([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 22, 33, 44, 55, 66, 77, 88, 99, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191])
        
        # Remove duplicates and filter valid palindromes
        palindromes = list(set(p for p in palindromes if 0 <= p < 10**9))
        
        # Calculate minimum cost
        min_cost = float('inf')
        for p in palindromes:
            cost = calculate_cost(nums, p)
            min_cost = min(min_cost, cost)
        
        return min_cost