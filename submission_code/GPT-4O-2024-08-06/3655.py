class Solution:
    def minOperations(self, n: int, m: int) -> int:
        def is_prime(x):
            if x <= 1:
                return False
            for i in range(2, int(x**0.5) + 1):
                if x % i == 0:
                    return False
            return True
        
        def transform(n, m):
            n_str = str(n)
            m_str = str(m)
            cost = 0
            current = n
            
            for i in range(len(n_str)):
                n_digit = int(n_str[i])
                m_digit = int(m_str[i])
                
                while n_digit != m_digit:
                    if n_digit < m_digit:
                        n_digit += 1
                    else:
                        n_digit -= 1
                    
                    current = int(n_str[:i] + str(n_digit) + n_str[i+1:])
                    
                    if is_prime(current):
                        return -1
                    
                    cost += current
                    n_str = str(current)
            
            return cost
        
        if is_prime(n):
            return -1
        
        return transform(n, m)