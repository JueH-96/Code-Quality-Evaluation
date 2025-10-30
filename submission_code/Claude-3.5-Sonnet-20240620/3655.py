class Solution:
    def minOperations(self, n: int, m: int) -> int:
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True

        def dfs(current, target, cost):
            if current == target:
                return cost
            
            if current > 9999 or current < 1:
                return float('inf')
            
            if is_prime(current):
                return float('inf')
            
            min_cost = float('inf')
            digits = list(str(current))
            
            for i in range(len(digits)):
                # Increase digit
                if int(digits[i]) < 9:
                    new_num = int(''.join(digits[:i] + [str(int(digits[i]) + 1)] + digits[i+1:]))
                    min_cost = min(min_cost, dfs(new_num, target, cost + new_num))
                
                # Decrease digit
                if int(digits[i]) > 0:
                    new_num = int(''.join(digits[:i] + [str(int(digits[i]) - 1)] + digits[i+1:]))
                    min_cost = min(min_cost, dfs(new_num, target, cost + new_num))
            
            return min_cost

        if is_prime(n) or is_prime(m):
            return -1

        result = dfs(n, m, n)
        return result if result != float('inf') else -1