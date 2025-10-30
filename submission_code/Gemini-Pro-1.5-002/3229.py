class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        ans = float('inf')

        def is_palindrome(num):
            s = str(num)
            return s == s[::-1]

        for i in range(n):
            cost = 0
            for j in range(n):
                cost += abs(nums[j] - nums[i])
            if is_palindrome(nums[i]):
                ans = min(ans, cost)
        
        
        for i in range(n // 2):
            
            l, r = 0, 100000
            
            
            while l <= r:
                mid = (l + r) // 2
                
                num = nums[i] + mid
                
                if is_palindrome(num):
                    
                    cost = 0
                    for j in range(n):
                        cost += abs(nums[j] - num)
                    ans = min(ans, cost)
                    break
                elif int(str(num)[::-1]) > num:
                    r = mid - 1
                else:
                    l = mid + 1
            
            l, r = 0, 100000
            while l <= r:
                mid = (l + r) // 2
                
                num = nums[i] - mid
                if num < 0:
                    break
                if is_palindrome(num):
                    
                    cost = 0
                    for j in range(n):
                        cost += abs(nums[j] - num)
                    ans = min(ans, cost)
                    break
                elif int(str(num)[::-1]) < num:
                    r = mid - 1
                else:
                    l = mid + 1

        return ans