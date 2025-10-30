class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if k == 2 or k == 5:
            if n == 1:
                return '8' if k == 2 else '5'
            else:
                mid = '9' * (n - 2)
                return ('8' + mid + '8') if k == 2 else ('5' + mid + '5')
        
        allowed_first = self.generate_allowed_first(k)
        m = (n + 1) // 2
        
        for d in allowed_first:
            left_part = d + '9' * (m - 1)
            while True:
                if not left_part or left_part[0] != d:
                    break
                palindrome = self.construct_palindrome(left_part, n)
                if self.is_divisible(palindrome, k):
                    return palindrome
                new_left = self.decrement(left_part)
                if new_left is None:
                    break
                left_part = new_left
        
        return ""
    
    def generate_allowed_first(self, k):
        if k == 1:
            return ['9', '8', '7', '6', '5', '4', '3', '2', '1']
        elif k == 2:
            return ['8', '6', '4', '2']
        elif k == 3:
            return ['9', '8', '7', '6', '5', '4', '3', '2', '1']
        elif k == 4:
            return ['8', '6', '4', '2']
        elif k == 5:
            return ['5']
        elif k == 6:
            return ['8', '6', '4', '2']
        elif k == 7:
            return ['9', '8', '7', '6', '5', '4', '3', '2', '1']
        elif k == 8:
            return ['8', '6', '4', '2']
        elif k == 9:
            return ['9', '8', '7', '6', '5', '4', '3', '2', '1']
    
    def construct_palindrome(self, left_part, n):
        if n % 2 == 0:
            right_part = left_part[::-1]
        else:
            right_part = left_part[:-1][::-1]
        return left_part + right_part
    
    def is_divisible(self, palindrome, k):
        mod = 0
        for c in palindrome:
            mod = (mod * 10 + int(c)) % k
        return mod == 0
    
    def decrement(self, s):
        s_list = list(s)
        i = len(s_list) - 1
        while i >= 0 and s_list[i] == '0':
            s_list[i] = '9'
            i -= 1
        if i < 0:
            return None
        new_digit = int(s_list[i]) - 1
        if new_digit < 0:
            return None
        s_list[i] = str(new_digit)
        if i == 0 and s_list[i] == '0':
            return None
        return ''.join(s_list)