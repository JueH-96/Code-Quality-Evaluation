class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        if n == 1 and k > 9:
            return 0
        if n == 1:
            return len([i for i in range(1,10) if i % k == 0])

        count = 0
        
        def is_palindrome(num_str):
            return num_str == num_str[::-1]

        def is_divisible(num,k):
            return num % k == 0

        def permute(nums, l, r):
            nonlocal count
            if l==r:
                num_str = "".join(map(str,nums))
                num = int(num_str)
                if is_palindrome(num_str) and is_divisible(num,k) and num_str[0] != '0':
                    
                    return
            else:
                for i in range(l,r+1):
                    nums[l], nums[i] = nums[i], nums[l]
                    permute(nums, l+1, r)
                    nums[l], nums[i] = nums[i], nums[l]

        
        digits = [0] * n
        
        def generate(index):
            nonlocal count
            if index == n:
                
                if digits[0] == 0:
                    return
                
                permute(digits[:],0,n-1)
                return

            for i in range(10):
                digits[index] = i
                generate(index + 1)

        generate(0)
        
        
        return count