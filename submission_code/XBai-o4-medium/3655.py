import heapq

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        if n == m:
            return n if not self.is_prime(n) else -1
        
        if self.is_prime(n) or self.is_prime(m):
            return -1
        
        def get_neighbors(num):
            s = list(str(num))
            digits = [int(c) for c in s]
            neighbors = []
            length = len(digits)
            for i in range(length):
                # Try increasing
                if digits[i] < 9:
                    new_digits = digits.copy()
                    new_digits[i] += 1
                    new_s = ''.join([str(d) for d in new_digits])
                    new_num = int(new_s)
                    neighbors.append(new_num)
                # Try decreasing
                if digits[i] > 0:
                    new_digits = digits.copy()
                    new_digits[i] -= 1
                    # Check for leading zero
                    if i == 0 and new_digits[i] == 0:
                        continue
                    new_s = ''.join([str(d) for d in new_digits])
                    new_num = int(new_s)
                    neighbors.append(new_num)
            return neighbors
        
        pq = []
        heapq.heappush(pq, (n, n))
        min_cost = {n: n}
        
        while pq:
            current_total, current_num = heapq.heappop(pq)
            if current_num == m:
                return current_total
            if current_total > min_cost.get(current_num, float('inf')):
                continue
            for neighbor in get_neighbors(current_num):
                if self.is_prime(neighbor):
                    continue
                new_total = current_total + neighbor
                if neighbor not in min_cost or new_total < min_cost[neighbor]:
                    min_cost[neighbor] = new_total
                    heapq.heappush(pq, (new_total, neighbor))
        
        return -1
        
    def is_prime(self, num: int) -> bool:
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