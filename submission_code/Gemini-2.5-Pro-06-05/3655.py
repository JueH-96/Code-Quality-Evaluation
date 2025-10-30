import heapq

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        
        MAX_VAL = 10000
        
        # 1. Sieve of Eratosthenes to find all non-prime (composite) numbers up to MAX_VAL.
        # This allows for O(1) checks for the "must not be a prime" constraint.
        is_composite = [False] * MAX_VAL
        is_composite[0] = True  # 0 is not prime
        is_composite[1] = True  # 1 is not prime
        for p in range(2, int(MAX_VAL**0.5) + 1):
            if not is_composite[p]:  # If p is prime
                # Mark all multiples of p as composite
                for i in range(p * p, MAX_VAL, p):
                    is_composite[i] = True
        
        # 2. Initial Checks
        # If n or m is prime, it's impossible to start or finish.
        if not is_composite[n] or not is_composite[m]:
            return -1
        
        # If n and m are the same, no operations are needed. The cost is just n.
        if n == m:
            return n

        # 3. Modified Dijkstra's algorithm to find the shortest path.
        # The "cost" or "distance" to a node is the sum of all node values on the path.
        dist = [float('inf')] * MAX_VAL
        
        # The cost to reach the starting node n is n itself.
        dist[n] = n
        
        # Priority queue stores tuples of (current_path_cost, number).
        pq = [(n, n)]
        
        while pq:
            current_cost, u = heapq.heappop(pq)
            
            # If we've found a better path to u already, skip.
            if current_cost > dist[u]:
                continue
            
            # If we've reached the target, return the accumulated cost.
            if u == m:
                return current_cost
            
            # 4. Generate neighbors by changing one digit at a time.
            s_u = str(u)
            num_digits = len(s_u)
            
            for i in range(num_digits):
                power_of_10 = 10**(num_digits - 1 - i)
                digit = (u // power_of_10) % 10

                # Operation: Increase digit
                if digit < 9:
                    v = u + power_of_10
                    # Check if the new number is valid (composite and within bounds).
                    if v < MAX_VAL and is_composite[v]:
                        # The cost of the new path is the old path's cost plus the new node's value.
                        new_cost_to_v = current_cost + v
                        if new_cost_to_v < dist[v]:
                            dist[v] = new_cost_to_v
                            heapq.heappush(pq, (new_cost_to_v, v))
                            
                # Operation: Decrease digit
                if digit > 0:
                    v = u - power_of_10
                    if v < MAX_VAL and is_composite[v]:
                        new_cost_to_v = current_cost + v
                        if new_cost_to_v < dist[v]:
                            dist[v] = new_cost_to_v
                            heapq.heappush(pq, (new_cost_to_v, v))

        # 5. If the queue is empty and we haven't reached m, it's unreachable.
        return -1