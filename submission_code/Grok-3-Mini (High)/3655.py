import math
import heapq

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        # Compute is_prime using Sieve of Eratosthenes
        max_num = 10000
        is_prime = [True] * (max_num + 1)
        is_prime[0] = False
        is_prime[1] = False
        for p in range(2, int(math.sqrt(max_num)) + 1):
            if is_prime[p]:
                for multiple in range(p * p, max_num + 1, p):
                    is_prime[multiple] = False
        
        # Check if n or m is prime
        if is_prime[n] or is_prime[m]:
            return -1
        
        # Define get_neighbors function
        def get_neighbors(node):
            s = str(node)
            len_s = len(s)
            neighbors = []
            for i in range(len_s):
                digit = int(s[i])
                # Try to increase the digit
                if digit < 9:
                    new_digit = digit + 1
                    new_str = s[:i] + str(new_digit) + s[i + 1:]
                    y = int(new_str)
                    if not is_prime[y]:  # y is not prime
                        neighbors.append(y)
                # Try to decrease the digit
                if digit > 0:
                    new_digit = digit - 1
                    new_str = s[:i] + str(new_digit) + s[i + 1:]
                    y = int(new_str)
                    if not is_prime[y]:  # y is not prime
                        neighbors.append(y)
            return neighbors
        
        # Dijkstra's algorithm
        INF = float('inf')
        dist = [INF] * (max_num + 1)
        dist[n] = n  # Initialize distance to start node
        heap = []
        heapq.heappush(heap, (dist[n], n))  # (distance, node)
        
        while heap:
            current_dist, current_node = heapq.heappop(heap)
            if current_dist > dist[current_node]:  # Skip outdated entries
                continue
            for neighbor in get_neighbors(current_node):
                new_dist = current_dist + neighbor  # Add weight of neighbor
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor))
        
        if dist[m] == INF:
            return -1
        else:
            return dist[m]