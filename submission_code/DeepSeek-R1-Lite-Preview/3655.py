class Solution:
    def minOperations(self, n: int, m: int) -> int:
        from heapq import heappush, heappop

        # Function to determine the number of digits
        def num_digits(x):
            return len(str(x))
        
        # Sieve of Eratosthenes to find primes up to 9999
        def sieve(limit):
            is_prime = [True] * (limit + 1)
            is_prime[0] = is_prime[1] = False
            for i in range(2, int(limit**0.5) + 1):
                if is_prime[i]:
                    for j in range(i*i, limit + 1, i):
                        is_prime[j] = False
            return is_prime
        
        # Precompute primes
        is_prime = sieve(9999)
        
        # Check if n or m is prime
        if is_prime[n] or is_prime[m]:
            return -1
        
        # Ensure n and m have the same number of digits
        if num_digits(n) != num_digits(m):
            return -1
        
        # Function to get all neighbors by changing one digit up or down
        def get_neighbors(current, num_digits):
            s = str(current).zfill(num_digits)
            neighbors = []
            for i in range(num_digits):
                digit = int(s[i])
                # Try decreasing the digit
                if digit > 0:
                    new_digit = digit - 1
                    new_s = s[:i] + str(new_digit) + s[i+1:]
                    new_num = int(new_s)
                    # Ensure the number has the same number of digits
                    if num_digits(new_num) == num_digits and not is_prime[new_num]:
                        neighbors.append(new_num)
                # Try increasing the digit
                if digit < 9:
                    new_digit = digit + 1
                    new_s = s[:i] + str(new_digit) + s[i+1:]
                    new_num = int(new_s)
                    # Ensure the number has the same number of digits
                    if num_digits(new_num) == num_digits and not is_prime[new_num]:
                        neighbors.append(new_num)
            return neighbors
        
        # Dijkstra's algorithm to find the minimal sum path
        def dijkstra(start, end, num_digits):
            heap = []
            heappush(heap, (start, start))  # (cumulative_sum, current_number)
            visited = {}
            visited[start] = start
            while heap:
                current_sum, current_num = heappop(heap)
                if current_num == end:
                    return current_sum
                if visited.get(current_num, float('inf')) < current_sum:
                    continue  # This is not the minimal sum to reach current_num
                neighbors = get_neighbors(current_num, num_digits)
                for neighbor in neighbors:
                    if neighbor not in visited or current_sum + neighbor < visited[neighbor]:
                        visited[neighbor] = current_sum + neighbor
                        heappush(heap, (current_sum + neighbor, neighbor))
            return -1
        
        digit_count = num_digits(n)
        return dijkstra(n, m, digit_count)