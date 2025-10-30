class Solution:
    def minOperations(self, n: int, m: int) -> int:
        # Check if a number is prime
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
        
        # If n or m is prime, it's impossible
        if is_prime(n) or is_prime(m):
            return -1
        
        # Get neighbors by changing one digit
        def get_neighbors(num):
            neighbors = []
            s = str(num)
            for i in range(len(s)):
                digit = int(s[i])
                # Try increasing the digit
                if digit < 9:
                    new_s = s[:i] + str(digit + 1) + s[i+1:]
                    new_num = int(new_s)
                    if not is_prime(new_num):
                        neighbors.append(new_num)
                # Try decreasing the digit
                if digit > 0:
                    new_s = s[:i] + str(digit - 1) + s[i+1:]
                    new_num = int(new_s)
                    if not is_prime(new_num):
                        neighbors.append(new_num)
            return neighbors
        
        # Dijkstra's algorithm
        import heapq
        
        # Priority queue: (cost, current_number)
        pq = [(n, n)]
        visited = set()
        
        while pq:
            cost, curr = heapq.heappop(pq)
            
            if curr in visited:
                continue
            
            visited.add(curr)
            
            if curr == m:
                return cost
            
            for neighbor in get_neighbors(curr):
                if neighbor not in visited:
                    heapq.heappush(pq, (cost + neighbor, neighbor))
        
        return -1