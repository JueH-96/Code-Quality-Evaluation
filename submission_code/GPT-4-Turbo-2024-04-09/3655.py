from collections import deque
import math

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        if n == m:
            return 0 if not self.is_prime(n) else -1
        
        if self.is_prime(m):
            return -1
        
        # Convert integers to strings for easier manipulation
        n_str = str(n)
        m_str = str(m)
        
        # BFS setup
        queue = deque([(n, 0, n)])  # (current_value, cost, sum_of_values)
        visited = set()
        visited.add(n)
        
        while queue:
            current, cost, sum_values = queue.popleft()
            
            if current == m:
                return sum_values
            
            # Generate all possible next states
            current_str = str(current)
            for i in range(len(current_str)):
                current_digit = int(current_str[i])
                
                # Try increasing the digit
                if current_digit < 9:
                    new_digit = current_digit + 1
                    new_value = self.replace_digit(current_str, i, new_digit)
                    if new_value not in visited and not self.is_prime(new_value):
                        visited.add(new_value)
                        queue.append((new_value, cost + 1, sum_values + new_value))
                
                # Try decreasing the digit
                if current_digit > 0:
                    new_digit = current_digit - 1
                    new_value = self.replace_digit(current_str, i, new_digit)
                    if new_value not in visited and not self.is_prime(new_value):
                        visited.add(new_value)
                        queue.append((new_value, cost + 1, sum_values + new_value))
        
        return -1
    
    def replace_digit(self, s, idx, new_digit):
        # Replace the digit at index idx with new_digit in string s
        return int(s[:idx] + str(new_digit) + s[idx+1:])
    
    def is_prime(self, num):
        if num <= 1:
            return False
        if num <= 3:
            return True
        if num % 2 == 0 or num % 3 == 0:
            return False
        i = 5
        while i * i <= num:
            if num % i == 0 or num % (i + 2) == 0:
                return False
            i += 6
        return True