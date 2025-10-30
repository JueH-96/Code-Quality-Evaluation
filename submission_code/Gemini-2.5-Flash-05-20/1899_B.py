import sys
import math

def solve():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))

    # Calculate prefix sums. P[i] stores the sum of elements from a[0] to a[i-1].
    # So, P[0] = 0, P[1] = a[0], P[2] = a[0] + a[1], etc.
    # The sum of a[start_idx] to a[end_idx-1] is P[end_idx] - P[start_idx].
    P = [0] * (n + 1)
    for i in range(n):
        P[i+1] = P[i] + a[i]

    max_overall_diff = 0

    # Iterate through possible values of k (number of boxes per truck).
    # A valid k must be a divisor of n.
    # We find divisors by iterating up to sqrt(n). If k_val divides n, then
    # both k_val and n // k_val are divisors.
    for k_val in range(1, int(math.isqrt(n)) + 1):
        if n % k_val == 0:
            current_divisors = [k_val]
            # If k_val is not the square root of n, then n // k_val is a distinct divisor.
            if k_val * k_val != n:
                current_divisors.append(n // k_val)
            
            for current_k in current_divisors:
                num_trucks = n // current_k

                # If there's only one truck, the problem states the difference is 0.
                # So we only proceed if there are at least two trucks.
                if num_trucks <= 1: 
                    continue

                # Initialize min_truck_weight and max_truck_weight with the weight of the first truck.
                # The first truck takes boxes from index 0 to current_k-1.
                # Its weight is P[current_k] - P[0].
                first_truck_weight = P[current_k] - P[0]
                min_truck_weight = first_truck_weight
                max_truck_weight = first_truck_weight
                
                # Calculate weights for subsequent trucks and update min/max.
                # Truck 'i' (0-indexed) takes boxes from index i*current_k to (i+1)*current_k - 1.
                # Its weight is P[(i+1)*current_k] - P[i*current_k].
                for i in range(1, num_trucks): # Start from the second truck (index 1)
                    start_idx_for_current_truck = i * current_k
                    end_idx_for_current_truck = (i + 1) * current_k
                    current_truck_weight = P[end_idx_for_current_truck] - P[start_idx_for_current_truck]
                    
                    if current_truck_weight < min_truck_weight:
                        min_truck_weight = current_truck_weight
                    if current_truck_weight > max_truck_weight:
                        max_truck_weight = current_truck_weight
                
                # The maximum absolute difference for this 'current_k' is max_truck_weight - min_truck_weight.
                # Update the overall maximum difference found so far.
                current_diff = max_truck_weight - min_truck_weight
                if current_diff > max_overall_diff:
                    max_overall_diff = current_diff
    
    # Print the final maximum overall difference.
    sys.stdout.write(str(max_overall_diff) + "
")

# Read the number of test cases.
num_test_cases = int(sys.stdin.readline())

# Process each test case.
for _ in range(num_test_cases):
    solve()