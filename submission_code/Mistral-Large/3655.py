from collections import deque
from sympy import isprime

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        if isprime(n) or isprime(m):
            return -1

        def get_neighbors(num):
            neighbors = []
            num_str = str(num)
            for i in range(len(num_str)):
                if num_str[i] != '9':
                    new_num = int(num_str[:i] + str(int(num_str[i]) + 1) + num_str[i+1:])
                    if not isprime(new_num):
                        neighbors.append(new_num)
                if num_str[i] != '0':
                    new_num = int(num_str[:i] + str(int(num_str[i]) - 1) + num_str[i+1:])
                    if not isprime(new_num):
                        neighbors.append(new_num)
            return neighbors

        queue = deque([(n, n)])
        visited = set([n])

        while queue:
            current, cost = queue.popleft()
            if current == m:
                return cost
            for neighbor in get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, cost + neighbor))

        return -1