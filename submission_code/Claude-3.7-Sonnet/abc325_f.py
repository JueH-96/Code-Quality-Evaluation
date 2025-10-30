# YOUR CODE HERE
import math

def min_cost_to_monitor(N, D, L1, C1, K1, L2, C2, K2):
    # Memoization cache
    memo = {}
    
    def dp(i, k1, k2):
        """
        Calculate the minimum cost to monitor sections i to N-1
        using at most k1 type-1 sensors and k2 type-2 sensors.
        """
        # Base case: all sections are monitored
        if i == N:
            return 0
        
        # Check memo cache
        if (i, k1, k2) in memo:
            return memo[(i, k1, k2)]
        
        result = float('inf')
        # Try all possible combinations of type-1 sensors
        max_type1 = min(k1, math.ceil(D[i] / L1))
        for type1 in range(max_type1 + 1):
            # Calculate remaining length to monitor after using type1 sensors
            remaining_length = max(0, D[i] - type1 * L1)
            # Calculate type-2 sensors needed for the remaining length
            type2_needed = math.ceil(remaining_length / L2) if remaining_length > 0 else 0
            
            # Skip if we need more type-2 sensors than available
            if type2_needed > k2:
                continue
            
            # Calculate cost for current section
            section_cost = type1 * C1 + type2_needed * C2
            # Calculate cost for remaining sections
            next_cost = dp(i + 1, k1 - type1, k2 - type2_needed)
            
            # If it's possible to monitor remaining sections
            if next_cost != float('inf'):
                # Update minimum cost
                result = min(result, section_cost + next_cost)
        
        # Store result in memo cache
        memo[(i, k1, k2)] = result
        return result
    
    # Start the DP calculation
    min_cost = dp(0, K1, K2)
    
    # Return -1 if it's impossible to monitor all sections
    return min_cost if min_cost != float('inf') else -1

# Read input
N = int(input())
D = list(map(int, input().split()))
L1, C1, K1 = map(int, input().split())
L2, C2, K2 = map(int, input().split())

# Calculate and print the result
print(min_cost_to_monitor(N, D, L1, C1, K1, L2, C2, K2))