class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        median = nums[n // 2]
        
        if self.is_palindrome(median):
            return sum(abs(num - median) for num in nums)
        else:
            prev_pal = self.previous_palindrome(median)
            next_pal = self.next_palindrome(median)
            cost_prev = sum(abs(num - prev_pal) for num in nums)
            cost_next = sum(abs(num - next_pal) for num in nums)
            return min(cost_prev, cost_next)
    
    def is_palindrome(self, x):
        s = str(x)
        return s == s[::-1]
    
    def next_palindrome(self, n):
        candidate = n
        while True:
            if self.is_palindrome(candidate):
                return candidate
            candidate += 1
    
    def previous_palindrome(self, n):
        if n == 0:
            return 0
        candidate = n
        while candidate >= 0:
            if self.is_palindrome(candidate):
                return candidate
            candidate -= 1
        return 0