class Solution:
    def minOperations(self, n: int, m: int) -> int:
        import heapq
        
        # Helper function to check if a number is prime
        def is_prime(num):
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
        
        # Check if n or m is prime
        if is_prime(n) or is_prime(m):
            return -1
        
        # Priority queue for Dijkstra's algorithm: (cost, current_n)
        pq = [(n, n)]
        visited = {n: n}  # node: cost
        
        while pq:
            cost, current_n = heapq.heappop(pq)
            
            if current_n in visited and visited[current_n] < cost:
                continue  # Already processed with a better cost
            
            if current_n == m:
                return cost
            
            # Explore all possible operations
            digits = [int(d) for d in str(current_n)]
            for i, digit in enumerate(digits):
                # Increase digit by 1
                if digit < 9:
                    new_digits = digits.copy()
                    new_digits[i] += 1
                    new_n = int(''.join(map(str, new_digits)))
                    
                    if not is_prime(new_n):
                        new_cost = cost + new_n
                        if new_n not in visited or new_cost < visited[new_n]:
                            visited[new_n] = new_cost
                            heapq.heappush(pq, (new_cost, new_n))
                
                # Decrease digit by 1
                if digit > 0:
                    new_digits = digits.copy()
                    new_digits[i] -= 1
                    new_n = int(''.join(map(str, new_digits)))
                    
                    if not is_prime(new_n):
                        new_cost = cost + new_n
                        if new_n not in visited or new_cost < visited[new_n]:
                            visited[new_n] = new_cost
                            heapq.heappush(pq, (new_cost, new_n))
        
        return -1