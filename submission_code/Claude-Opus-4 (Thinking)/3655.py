class Solution:
    def minOperations(self, n: int, m: int) -> int:
        import heapq
        
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
        
        # Check if transformation is possible
        if is_prime(n) or is_prime(m):
            return -1
        
        # Dijkstra's algorithm for minimum cost
        num_digits = len(str(n))
        pq = [(n, n)]  # (cost, current_number)
        visited = set()
        costs = {n: n}
        
        while pq:
            cost, current = heapq.heappop(pq)
            
            if current == m:
                return cost
            
            if current in visited:
                continue
            visited.add(current)
            
            # Try changing each digit
            str_current = str(current)
            
            for i in range(num_digits):
                digit = int(str_current[i])
                
                # Increase digit by 1 (if not 9)
                if digit < 9:
                    new_str = str_current[:i] + str(digit + 1) + str_current[i+1:]
                    new_num = int(new_str)
                    if not is_prime(new_num) and new_num not in visited:
                        new_cost = cost + new_num
                        if new_num not in costs or costs[new_num] > new_cost:
                            costs[new_num] = new_cost
                            heapq.heappush(pq, (new_cost, new_num))
                
                # Decrease digit by 1 (if not 0)
                if digit > 0:
                    new_str = str_current[:i] + str(digit - 1) + str_current[i+1:]
                    new_num = int(new_str)
                    # Ensure we maintain same number of digits
                    if len(str(new_num)) == num_digits and not is_prime(new_num) and new_num not in visited:
                        new_cost = cost + new_num
                        if new_num not in costs or costs[new_num] > new_cost:
                            costs[new_num] = new_cost
                            heapq.heappush(pq, (new_cost, new_num))
        
        return -1