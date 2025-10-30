import heapq
import math

# Precomputation (outside the class)
# Sieve of Eratosthenes to find primes up to MAX_VAL
MAX_VAL = 10000
is_prime = [True] * (MAX_VAL + 1)
is_prime[0] = is_prime[1] = False
# Start from 2 and mark multiples as not prime
for p in range(2, int(math.sqrt(MAX_VAL)) + 1):
    if is_prime[p]:
        # Mark multiples of p starting from p*p
        for i in range(p*p, MAX_VAL + 1, p):
            is_prime[i] = False

# Helper function to check if a number is valid (composite or 1)
def is_composite_or_one(num):
    # A number is valid in our context if it's 1 or it's greater than 1 and not prime
    return num == 1 or (num > 1 and not is_prime[num])

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        # Initial check: The starting number must not be prime (unless it's 1)
        if not is_composite_or_one(n):
            return -1

        # Determine the range of numbers based on the number of digits of n (and m)
        s_n = str(n)
        num_digits = len(s_n)
        
        # Calculate the minimum and maximum values for numbers with num_digits
        if num_digits == 1:
            min_val = 1
            max_val = 9
        else:
            min_val = 10**(num_digits - 1)
            max_val = 10**num_digits - 1

        # Use a dictionary to store the minimum cumulative cost to reach each number
        # Initialize dist with infinity for all numbers except n
        dist = {}
        
        # Priority queue to implement Dijkstra's. Stores tuples (cumulative_cost, current_number)
        pq = []

        # Initialize the starting state: cost to reach n is n itself
        dist[n] = n
        heapq.heappush(pq, (n, n))

        # Dijkstra's algorithm begins
        while pq:
            # Get the number with the smallest cumulative cost found so far
            current_cost, current_num = heapq.heappop(pq)

            # If we found a cheaper path to current_num already, skip this one
            # This check is necessary because we can push multiple entries for the same number
            # into the heap if we find new paths, and we only want to process the one
            # with the minimum cost when it's popped.
            if current_num in dist and current_cost > dist[current_num]:
                 continue

            # If we reached the target number m, return the cumulative cost
            if current_num == m:
                return current_cost

            # Generate possible next numbers by applying one operation (increase or decrease a digit)
            s_current = str(current_num)
            
            # Iterate through each digit of the current number
            for i in range(num_digits):
                digit = int(s_current[i])

                # --- Try increasing the digit ---
                if digit < 9:
                    # Create the new number string by increasing the digit at index i
                    new_s_list = list(s_current)
                    new_s_list[i] = str(digit + 1)
                    next_num = int("".join(new_s_list))

                    # Check if the generated number is within the valid range [min_val, max_val]
                    # This implicitly checks if the number of digits is preserved
                    # Also check if the generated number is a valid state (composite or 1)
                    if next_num >= min_val and next_num <= max_val and is_composite_or_one(next_num):
                        # Calculate the cumulative cost to reach next_num via the current path
                        new_cost = current_cost + next_num

                        # If this path is better than any previously found path to next_num
                        # or if next_num is visited for the first time
                        if next_num not in dist or new_cost < dist[next_num]:
                            dist[next_num] = new_cost
                            # Push the new state (cost, number) to the priority queue
                            heapq.heappush(pq, (new_cost, next_num))

                # --- Try decreasing the digit ---
                if digit > 0:
                    # Create the new number string by decreasing the digit at index i
                    new_s_list = list(s_current)
                    new_s_list[i] = str(digit - 1)
                    next_num = int("".join(new_s_list))

                    # Check if the generated number is within the valid range [min_val, max_val]
                    # This implicitly checks if the number of digits is preserved and prevents leading zeros for multi-digit numbers
                    # Also check if the generated number is a valid state (composite or 1)
                    if next_num >= min_val and next_num <= max_val and is_composite_or_one(next_num):
                        # Calculate the cumulative cost to reach next_num via the current path
                        new_cost = current_cost + next_num

                        # If this path is better than any previously found path to next_num
                        # or if next_num is visited for the first time
                        if next_num not in dist or new_cost < dist[next_num]:
                            dist[next_num] = new_cost
                            # Push the new state (cost, number) to the priority queue
                            heapq.heappush(pq, (new_cost, next_num))

        # If the priority queue becomes empty and the target m was not reached
        return -1