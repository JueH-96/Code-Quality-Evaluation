import heapq

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        if n == m and not self.is_prime(n):
            return n
        if self.is_prime(n) or self.is_prime(m):
            return -1
        
        k = len(str(n))
        visited = set()
        min_sum = {n: n}
        pq = [(n, n)]  # (current_sum, current_number)
        
        while pq:
            current_sum, current = heapq.heappop(pq)
            if current == m:
                return current_sum
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.get_neighbors(current, k):
                new_sum = current_sum + neighbor
                if neighbor not in min_sum or new_sum < min_sum[neighbor]:
                    min_sum[neighbor] = new_sum
                    heapq.heappush(pq, (new_sum, neighbor))
        return -1
    
    def is_prime(self, num):
        if num < 2:
            return False
        return self.primes[num]
    
    def get_neighbors(self, num, k):
        num_str = f"{num:0{k}d}"
        neighbors = []
        for i in range(k):
            digit = num_str[i]
            if i == 0:
                if digit != '9':
                    new_str = num_str[:i] + str(int(digit) + 1) + num_str[i+1:]
                    neighbor = int(new_str)
                    if not self.is_prime(neighbor):
                        neighbors.append(neighbor)
                if digit > '1':
                    new_str = num_str[:i] + str(int(digit) - 1) + num_str[i+1:]
                    neighbor = int(new_str)
                    if not self.is_prime(neighbor):
                        neighbors.append(neighbor)
            else:
                if digit != '9':
                    new_str = num_str[:i] + str(int(digit) + 1) + num_str[i+1:]
                    neighbor = int(new_str)
                    if not self.is_prime(neighbor):
                        neighbors.append(neighbor)
                if digit != '0':
                    new_str = num_str[:i] + str(int(digit) - 1) + num_str[i+1:]
                    neighbor = int(new_str)
                    if not self.is_prime(neighbor):
                        neighbors.append(neighbor)
        return neighbors
    
    def __init__(self):
        self.primes = [False] * 10000
        self.primes[0] = self.primes[1] = False
        for i in range(2, int(10000 ** 0.5) + 1):
            if self.primes[i]:
                for j in range(i * i, 10000, i):
                    self.primes[j] = False
        for i in range(2, 10000):
            if self.primes[i]:
                self.primes[i] = True
            else:
                self.primes[i] = False