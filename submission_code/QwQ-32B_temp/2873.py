from typing import List

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        if n < 2:
            return []
        is_prime = [True] * (n + 1)
        is_prime[0], is_prime[1] = False, False
        
        for num in range(2, int(n**0.5) + 1):
            if is_prime[num]:
                for multiple in range(num*num, n+1, num):
                    is_prime[multiple] = False
        
        result = []
        max_x = n // 2
        for x in range(2, max_x + 1):
            y = n - x
            if is_prime[x] and is_prime[y]:
                result.append([x, y])
        
        return result