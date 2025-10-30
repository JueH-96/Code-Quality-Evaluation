import heapq

class Solution:
    def __init__(self):
        max_limit = 9999
        self.is_prime = [True] * (max_limit + 1)
        self.is_prime[0] = self.is_prime[1] = False
        for i in range(2, int(max_limit ** 0.5) + 1):
            if self.is_prime[i]:
                for j in range(i * i, max_limit + 1, i):
                    self.is_prime[j] = False

    def minOperations(self, n: int, m: int) -> int:
        if n == m:
            return n if not self.is_prime[n] else -1
        if self.is_prime[n] or self.is_prime[m]:
            return -1
        
        def get_neighbors(num):
            s = list(str(num))
            neighbors = []
            for i in range(len(s)):
                digit = int(s[i])
                # Increment
                if digit < 9:
                    new_s = s.copy()
                    new_s[i] = str(digit + 1)
                    new_num = int(''.join(new_s))
                    neighbors.append(new_num)
                # Decrement
                if digit > 0:
                    new_s = s.copy()
                    new_s[i] = str(digit - 1)
                    new_num = int(''.join(new_s))
                    # Check leading zero
                    if new_s[0] != '0':
                        neighbors.append(new_num)
            return neighbors

        heap = [(n, n)]
        cost = {n: n}
        
        while heap:
            current_cost, current = heapq.heappop(heap)
            if current == m:
                return current_cost
            if current_cost > cost.get(current, float('inf')):
                continue
            neighbors = get_neighbors(current)
            for neighbor in neighbors:
                if not self.is_prime[neighbor]:
                    new_cost = current_cost + neighbor
                    if new_cost < cost.get(neighbor, float('inf')):
                        cost[neighbor] = new_cost
                        heapq.heappush(heap, (new_cost, neighbor))
        return -1