class Solution:
    def is_prime(self, num: int) -> bool:
        if num <= 1:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    def minOperations(self, n: int, m: int) -> int:
        if self.is_prime(m):
            return -1
        
        total_cost = 0
        current_value = n
        
        while current_value != m:
            if self.is_prime(current_value):
                return -1
            
            total_cost += current_value
            
            if current_value < m:
                # Increase digits
                for i in range(len(str(current_value))):
                    if current_value < m:
                        digit = (current_value // (10 ** i)) % 10
                        if digit < 9:
                            current_value += 10 ** i
            else:
                # Decrease digits
                for i in range(len(str(current_value))):
                    if current_value > m:
                        digit = (current_value // (10 ** i)) % 10
                        if digit > 0:
                            current_value -= 10 ** i
        
        total_cost += current_value  # Add the final value
        return total_cost