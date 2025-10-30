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
        
        # If target is prime, impossible
        if is_prime(m):
            return -1
        
        # If start is prime, impossible
        if is_prime(n):
            return -1
        
        # If already equal
        if n == m:
            return n
        
        import heapq
        
        # Dijkstra's algorithm
        pq = [(n, n)]  # (cost, current_number)
        visited = set()
        
        while pq:
            cost, curr = heapq.heappop(pq)
            
            if curr in visited:
                continue
            visited.add(curr)
            
            if curr == m:
                return cost
            
            # Try all possible operations
            curr_str = str(curr)
            
            for i in range(len(curr_str)):
                digit = int(curr_str[i])
                
                # Increase digit (if not 9)
                if digit < 9:
                    new_num_str = curr_str[:i] + str(digit + 1) + curr_str[i+1:]
                    new_num = int(new_num_str)
                    
                    if new_num not in visited and not is_prime(new_num):
                        heapq.heappush(pq, (cost + new_num, new_num))
                
                # Decrease digit (if not 0)
                if digit > 0:
                    new_num_str = curr_str[:i] + str(digit - 1) + curr_str[i+1:]
                    new_num = int(new_num_str)
                    
                    if new_num not in visited and not is_prime(new_num):
                        heapq.heappush(pq, (cost + new_num, new_num))
        
        return -1