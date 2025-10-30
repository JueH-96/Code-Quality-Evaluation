class Solution:
    def minOperations(self, n: int, m: int) -> int:
        # Helper function to check if a number is prime.
        def is_prime(x: int) -> bool:
            if x < 2:
                return False
            if x == 2:
                return True
            if x % 2 == 0:
                return False
            # Check up to sqrt(x)
            r = int(x ** 0.5)
            for i in range(3, r+1, 2):
                if x % i == 0:
                    return False
            return True

        # Convert number to string with fixed width (same length as the original n)
        digits = len(str(n))
        
        # initial and target state must be non-prime, else transformation is impossible.
        if is_prime(n) or is_prime(m):
            return -1

        from heapq import heappush, heappop
        
        # Dijkstra's algorithm initialization
        # cost accumulates the sum of all visited numbers (including initial and final state values)
        # State: (current_total_cost, current_state_int)
        heap = []
        heappush(heap, (n, n))
        # visited cost dict: state integer -> lowest cost found so far.
        best = {n: n}

        while heap:
            cost, curr = heappop(heap)
            # if current state is m, return the cost
            if curr == m:
                return cost
            # If we already found a better path, we skip.
            if cost > best[curr]:
                continue
            
            # Represent current state as zero-padded string.
            s = str(curr).zfill(digits)
            s_list = list(s)
            for i in range(len(s_list)):
                digit = int(s_list[i])
                # For each possible operation:
                # Increase if digit != 9.
                if digit < 9:
                    new_digit = digit + 1
                    new_s_list = s_list[:]
                    new_s_list[i] = str(new_digit)
                    new_str = "".join(new_s_list)
                    new_val = int(new_str)
                    # Only consider states that have the same number of digits (leading zeros allowed) and are not prime.
                    # Here we check is_prime on new_val.
                    if not is_prime(new_val):
                        new_cost = cost + new_val
                        if new_val not in best or new_cost < best[new_val]:
                            best[new_val] = new_cost
                            heappush(heap, (new_cost, new_val))
                # Decrease if digit != 0.
                if digit > 0:
                    new_digit = digit - 1
                    new_s_list = s_list[:]
                    new_s_list[i] = str(new_digit)
                    new_str = "".join(new_s_list)
                    new_val = int(new_str)
                    if not is_prime(new_val):
                        new_cost = cost + new_val
                        if new_val not in best or new_cost < best[new_val]:
                            best[new_val] = new_cost
                            heappush(heap, (new_cost, new_val))
        return -1