class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(n):
            return str(n) == str(n)[::-1]

        def calculate_cost(nums, target):
            cost = 0
            for num in nums:
                cost += abs(num - target)
            return cost

        palindromes = []
        for i in range(1, 100001):
            s = str(i)
            palindromes.append(int(s + s[::-1][1:]))
            palindromes.append(int(s + s[::-1]))
        
        palindromes = sorted(list(set(palindromes)))
        palindromes = [p for p in palindromes if p < 10**9]

        nums.sort()
        n = len(nums)
        median1 = nums[n // 2]
        median2 = nums[(n - 1) // 2]
        
        min_cost = float('inf')

        
        
        
        left = 0
        right = len(palindromes) -1
        
        
        while left <= right:
            mid = (left + right) // 2
            if palindromes[mid] < median1:
                left = mid + 1
            else:
                right = mid - 1
        
        
        for i in range(max(0,left-2), min(len(palindromes), left+3)):
            min_cost = min(min_cost, calculate_cost(nums, palindromes[i]))
        
        left = 0
        right = len(palindromes) -1
        
        while left <= right:
            mid = (left + right) // 2
            if palindromes[mid] < median2:
                left = mid + 1
            else:
                right = mid - 1
        
        for i in range(max(0,left-2), min(len(palindromes), left+3)):
            min_cost = min(min_cost, calculate_cost(nums, palindromes[i]))
        
        
        return min_cost