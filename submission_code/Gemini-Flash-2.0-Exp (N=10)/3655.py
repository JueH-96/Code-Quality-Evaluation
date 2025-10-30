import heapq

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True

        if is_prime(m):
            return -1
        
        q = [(0, n)]
        visited = {n}
        
        while q:
            cost, curr = heapq.heappop(q)
            
            if curr == m:
                return cost
            
            s_curr = str(curr)
            for i in range(len(s_curr)):
                digit = int(s_curr[i])
                
                # Increase digit
                if digit < 9:
                    new_digit = digit + 1
                    new_num = int(s_curr[:i] + str(new_digit) + s_curr[i+1:])
                    if not is_prime(new_num) and new_num not in visited:
                        visited.add(new_num)
                        heapq.heappush(q, (cost + new_num, new_num))
                
                # Decrease digit
                if digit > 0:
                    new_digit = digit - 1
                    new_num = int(s_curr[:i] + str(new_digit) + s_curr[i+1:])
                    if not is_prime(new_num) and new_num not in visited:
                        visited.add(new_num)
                        heapq.heappush(q, (cost + new_num, new_num))
        
        return -1