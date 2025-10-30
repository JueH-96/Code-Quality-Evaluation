from heapq import heappop, heappush

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        max_num = 10**4
        sieve = self.generate_sieve(max_num)
        
        if sieve[n] or sieve[m]:
            return -1
        
        str_n = str(n)
        str_m = str(m)
        if len(str_n) != len(str_m):
            return -1
        
        digits_length = len(str_n)
        if n == m:
            return n
        
        heap = []
        heappush(heap, (n, n))
        visited = {n: n}
        
        while heap:
            current_cost, current_num = heappop(heap)
            if current_num in visited and current_cost > visited[current_num]:
                continue
            if current_num == m:
                return current_cost
            
            neighbors = self.generate_neighbors(current_num, sieve, digits_length)
            for neighbor in neighbors:
                new_cost = current_cost + neighbor
                if neighbor not in visited or new_cost < visited.get(neighbor, float('inf')):
                    visited[neighbor] = new_cost
                    heappush(heap, (new_cost, neighbor))
        
        return -1
    
    def generate_sieve(self, max_num):
        sieve = [True] * (max_num + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(max_num**0.5) + 1):
            if sieve[i]:
                sieve[i*i : max_num + 1 : i] = [False] * len(sieve[i*i : max_num + 1 : i])
        return sieve
    
    def generate_neighbors(self, current_num, sieve, digits_length):
        s = str(current_num)
        neighbors = []
        for i in range(digits_length):
            d = int(s[i])
            
            # Increase digit
            if d != 9:
                new_d = d + 1
                new_s = s[:i] + str(new_d) + s[i+1:]
                if new_s[0] == '0':
                    continue
                new_num = int(new_s)
                if len(str(new_num)) == digits_length and not sieve[new_num]:
                    neighbors.append(new_num)
            
            # Decrease digit
            if d != 0:
                new_d = d - 1
                new_s = s[:i] + str(new_d) + s[i+1:]
                if new_s[0] == '0':
                    continue
                new_num = int(new_s)
                if len(str(new_num)) == digits_length and not sieve[new_num]:
                    neighbors.append(new_num)
        
        return neighbors