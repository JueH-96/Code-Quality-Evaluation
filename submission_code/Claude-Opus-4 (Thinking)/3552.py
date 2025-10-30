class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if k == 1 or k == 3 or k == 9:
            return '9' * n
        
        if k == 2:
            if n == 1:
                return '8'
            return '8' + '9' * (n - 2) + '8'
        
        if k == 5:
            if n == 1:
                return '5'
            return '5' + '9' * (n - 2) + '5'
        
        if k == 4:
            if n == 1:
                return '8'
            
            # Find best first two digits
            for d1 in range(9, 0, -1):
                for d2 in range(9, -1, -1):
                    if (10 * d2 + d1) % 4 == 0:
                        if n == 2:
                            return str(d1) * 2
                        elif n == 3:
                            return str(d1) + str(d2) + str(d1)
                        else:
                            mid = '9' * (n - 4)
                            return str(d1) + str(d2) + mid + str(d2) + str(d1)
        
        if k == 6:
            if n == 1:
                return '6'
            
            # Try even first digits in decreasing order
            for d1 in [8, 6, 4, 2]:
                if n == 2:
                    if (2 * d1) % 3 == 0:
                        return str(d1) * 2
                else:
                    middle_count = n - 2
                    total = 2 * d1 + 9 * middle_count
                    rem = total % 3
                    
                    if rem == 0:
                        return str(d1) + '9' * middle_count + str(d1)
                    elif rem == 1 and middle_count > 0:
                        if middle_count % 2 == 1:
                            mid = middle_count // 2
                            return str(d1) + '9' * mid + '8' + '9' * mid + str(d1)
                        else:
                            mid = middle_count // 2
                            return str(d1) + '9' * (mid - 1) + '88' + '9' * (mid - 1) + str(d1)
                    elif rem == 2 and middle_count > 0:
                        if middle_count % 2 == 1:
                            mid = middle_count // 2
                            return str(d1) + '9' * mid + '7' + '9' * mid + str(d1)
                        else:
                            mid = middle_count // 2
                            return str(d1) + '9' * (mid - 1) + '88' + '9' * (mid - 1) + str(d1)
        
        if k == 8:
            if n == 1:
                return '8'
            elif n == 2:
                return '88'
            
            # Find best first three digits
            for d1 in range(9, 0, -1):
                for d2 in range(9, -1, -1):
                    for d3 in range(9, -1, -1):
                        # Check based on n
                        if n == 3:
                            # Palindrome is d1 d2 d1, last 3 digits form 100*d1 + 10*d2 + d1
                            if d1 == d3 and (101 * d1 + 10 * d2) % 8 == 0:
                                return str(d1) + str(d2) + str(d1)
                        else:
                            # General case: last 3 digits are d3 d2 d1
                            if (100 * d3 + 10 * d2 + d1) % 8 == 0:
                                if n == 4:
                                    return str(d1) + str(d2) * 2 + str(d1)
                                elif n == 5:
                                    return str(d1) + str(d2) + str(d3) + str(d2) + str(d1)
                                else:
                                    mid = '9' * (n - 6)
                                    return str(d1) + str(d2) + str(d3) + mid + str(d3) + str(d2) + str(d1)
        
        if k == 7:
            if n == 1:
                return '7'
            
            # General approach
            half_len = (n + 1) // 2
            
            for num in range(10 ** half_len - 1, -1, -1):
                half = str(num).zfill(half_len)
                
                if half[0] == '0':
                    continue
                
                if n % 2 == 1:
                    pal = half + half[-2::-1]
                else:
                    pal = half + half[::-1]
                
                if int(pal) % 7 == 0:
                    return pal
        
        return ""