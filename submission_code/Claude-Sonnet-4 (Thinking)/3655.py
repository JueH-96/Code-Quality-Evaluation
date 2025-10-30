import heapq

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        def is_prime(num):
            if num < 2:
                return False
            if num == 2:
                return True
            if num % 2 == 0:
                return False
            for i in range(3, int(num**0.5) + 1, 2):
                if num % i == 0:
                    return False
            return True
        
        # Check if n or m is prime
        if is_prime(n) or is_prime(m):
            return -1
        
        if n == m:
            return n
        
        # Check if n and m have the same number of digits
        if len(str(n)) != len(str(m)):
            return -1
        
        target_digits = len(str(n))
        
        # Dijkstra's algorithm
        pq = [(n, n)]  # (cost, current_value)
        visited = set()
        
        while pq:
            cost, curr = heapq.heappop(pq)
            
            if curr in visited:
                continue
            visited.add(curr)
            
            if curr == m:
                return cost
            
            # Generate next states
            curr_str = str(curr)
            for i in range(len(curr_str)):
                digit = int(curr_str[i])
                
                # Increase digit by 1
                if digit < 9:
                    new_digit = digit + 1
                    new_str = curr_str[:i] + str(new_digit) + curr_str[i+1:]
                    new_val = int(new_str)
                    # Check if new value has the same number of digits
                    if len(str(new_val)) == target_digits and not is_prime(new_val) and new_val not in visited:
                        heapq.heappush(pq, (cost + new_val, new_val))
                
                # Decrease digit by 1
                if digit > 0:
                    new_digit = digit - 1
                    new_str = curr_str[:i] + str(new_digit) + curr_str[i+1:]
                    new_val = int(new_str)
                    # Check if new value has the same number of digits
                    if len(str(new_val)) == target_digits and not is_prime(new_val) and new_val not in visited:
                        heapq.heappush(pq, (cost + new_val, new_val))
        
        return -1