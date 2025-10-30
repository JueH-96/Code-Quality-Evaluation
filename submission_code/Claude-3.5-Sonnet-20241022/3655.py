class Solution:
    def minOperations(self, n: int, m: int) -> int:
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num ** 0.5) + 1):
                if num % i == 0:
                    return False
            return True
        
        # If target is prime or initial number is prime, return -1
        if is_prime(m) or is_prime(n):
            return -1
        
        # Convert numbers to strings to get number of digits
        str_n = str(n)
        str_m = str(m)
        
        # If numbers have different lengths, return -1
        if len(str_n) != len(str_m):
            return -1
        
        from collections import deque
        visited = set()
        
        # Queue stores (current number, cost so far)
        queue = deque([(n, n)])
        visited.add(n)
        
        while queue:
            curr_num, cost = queue.popleft()
            
            if curr_num == m:
                return cost
            
            # Convert current number to list of digits
            digits = list(str(curr_num))
            
            # Try all possible operations
            for i in range(len(digits)):
                # Try increasing digit if not 9
                if digits[i] != '9':
                    new_digits = digits.copy()
                    new_digits[i] = str(int(new_digits[i]) + 1)
                    new_num = int(''.join(new_digits))
                    
                    if new_num not in visited and not is_prime(new_num):
                        visited.add(new_num)
                        queue.append((new_num, cost + new_num))
                
                # Try decreasing digit if not 0
                if digits[i] != '0':
                    new_digits = digits.copy()
                    new_digits[i] = str(int(new_digits[i]) - 1)
                    new_num = int(''.join(new_digits))
                    
                    if new_num not in visited and not is_prime(new_num):
                        visited.add(new_num)
                        queue.append((new_num, cost + new_num))
        
        return -1