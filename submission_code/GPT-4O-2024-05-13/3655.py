class Solution:
    def minOperations(self, n: int, m: int) -> int:
        from itertools import permutations
        import math

        def is_prime(x):
            if x <= 1:
                return False
            if x <= 3:
                return True
            if x % 2 == 0 or x % 3 == 0:
                return False
            i = 5
            while i * i <= x:
                if x % i == 0 or x % (i + 2) == 0:
                    return False
                i += 6
            return True

        def get_neighbors(x):
            s = str(x)
            neighbors = set()
            for i in range(len(s)):
                if s[i] != '9':
                    neighbors.add(int(s[:i] + str(int(s[i]) + 1) + s[i+1:]))
                if s[i] != '0':
                    neighbors.add(int(s[:i] + str(int(s[i]) - 1) + s[i+1:]))
            return neighbors

        if is_prime(n):
            return -1

        queue = [(n, 0, n)]
        visited = set()
        visited.add(n)

        while queue:
            current, cost, total_cost = queue.pop(0)
            if current == m:
                return total_cost
            for neighbor in get_neighbors(current):
                if neighbor not in visited and not is_prime(neighbor):
                    visited.add(neighbor)
                    queue.append((neighbor, cost + 1, total_cost + neighbor))

        return -1