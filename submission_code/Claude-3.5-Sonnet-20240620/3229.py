class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(num):
            return str(num) == str(num)[::-1]

        def next_palindrome(num):
            num += 1
            while not is_palindrome(num):
                num += 1
            return num

        def prev_palindrome(num):
            num -= 1
            while not is_palindrome(num):
                num -= 1
            return num

        min_num, max_num = min(nums), max(nums)
        
        lower = prev_palindrome(min_num)
        upper = next_palindrome(max_num)
        
        current = lower
        min_cost = float('inf')
        
        while current <= upper:
            cost = sum(abs(num - current) for num in nums)
            min_cost = min(min_cost, cost)
            current = next_palindrome(current)
        
        return min_cost