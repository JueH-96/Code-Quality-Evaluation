class Solution:
    def __init__(self):
        self.max_num = 10**4
        self.sieve = self.sieve_of_eratosthenes(self.max_num)

    def sieve_of_eratosthenes(self, limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return sieve

    def minOperations(self, n: int, m: int) -> int:
        s_n = str(n)
        s_m = str(m)
        if len(s_n) != len(s_m):
            return -1
        L = len(s_n)
        
        if self.sieve[n] or self.sieve[m]:
            return -1
        if n == m:
            return n
        
        import heapq
        distance = [float('inf')] * (self.max_num + 1)
        distance[n] = n
        heap = []
        heapq.heappush(heap, (n, n))
        
        while heap:
            current_total, u = heapq.heappop(heap)
            if u == m:
                return current_total
            if current_total > distance[u]:
                continue
            neighbors = self.get_neighbors(u, L)
            for v in neighbors:
                if v > self.max_num:
                    continue
                new_total = current_total + v
                if new_total < distance[v]:
                    distance[v] = new_total
                    heapq.heappush(heap, (new_total, v))
        
        return -1

    def get_neighbors(self, u: int, L: int):
        digits = list(map(int, str(u)))
        neighbors = []
        for i in range(L):
            d = digits[i]
            # Try increasing the digit
            if d < 9:
                new_digits = digits.copy()
                new_digits[i] = d + 1
                new_num = int(''.join(map(str, new_digits)))
                if new_num < 10 ** (L - 1):
                    continue
                if self.sieve[new_num]:
                    continue
                neighbors.append(new_num)
            # Try decreasing the digit
            if d > 0:
                new_digits = digits.copy()
                new_digits[i] = d - 1
                new_num = int(''.join(map(str, new_digits)))
                if new_num < 10 ** (L - 1):
                    continue
                if self.sieve[new_num]:
                    continue
                neighbors.append(new_num)
        return neighbors