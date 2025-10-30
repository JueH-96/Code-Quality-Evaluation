# YOUR CODE HERE
import math
import sys

# Function to find all divisors k of n such that 1 <= k < n
def get_divisors(n):
    """
    Finds all divisors k of n such that 1 <= k < n.
    Returns a list of unique divisors. Excludes n itself.
    Handles n=1 correctly by returning an empty list.
    """
    divs = set()
    # Iterate from 1 up to floor(sqrt(n))
    # This loop finds divisors i and pairs n//i
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            # i is a divisor. Add it if it's less than n.
            # Since i <= sqrt(n), this check is only relevant for n=1 where i=1.
            # For n=1, 1 < 1 is false, so 1 is not added.
            if i < n: 
               divs.add(i)
            
            # n // i is also a divisor. Add it if it's less than n.
            # This check correctly excludes the divisor n itself (when i=1).
            if n // i < n: 
               divs.add(n // i)

    # Convert the set of divisors to a list for iteration. The order does not matter.
    return list(divs)

def solve():
    """
    Solves a single test case.
    Reads the number of boxes n and their weights a.
    Computes prefix sums for efficient calculation of truck weights.
    Finds all valid divisors k of n (where k < n, meaning more than one truck).
    For each valid k:
      Calculates the total weight of boxes for each of the n/k trucks.
      Finds the maximum absolute difference between the total weights of any two trucks.
    Keeps track of the overall maximum difference found across all valid k.
    Prints the final maximum difference.
    """
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))

    # Compute prefix sums
    # prefix_sum[p] will store the sum of the first p elements of a (i.e., a[0]...a[p-1])
    prefix_sum = [0] * (n + 1)
    for i in range(n):
        prefix_sum[i+1] = prefix_sum[i] + a[i]

    # Find all divisors k of n such that k < n.
    # If k=n, there's only one truck, and the difference is 0.
    # We initialize max_overall_diff to 0, so we only need to consider k < n.
    divisors = get_divisors(n)
    
    # Initialize the maximum difference found so far. If no divisors k < n exist (e.g., n=1), this will be the answer.
    max_overall_diff = 0

    # Iterate through each valid divisor k. A divisor k < n guarantees at least 2 trucks.
    for k in divisors:
        
        # Number of trucks for this k
        num_trucks = n // k
        
        # Calculate the weight of the first truck to initialize min/max tracking.
        # Truck i (1-based index) corresponds to boxes from index (i-1)*k to i*k - 1 (0-based indexing).
        # Its weight W_i = sum(a[j] for j in range((i-1)*k, i*k))
        # Using prefix sums: W_i = prefix_sum[i*k] - prefix_sum[(i-1)*k]
        
        first_W = prefix_sum[k] - prefix_sum[0]
        current_max_W = first_W
        current_min_W = first_W
        
        # Iterate through the remaining trucks (from the 2nd truck up to the last one)
        for i in range(2, num_trucks + 1):
            # Calculate weight of the i-th truck
            W_i = prefix_sum[i*k] - prefix_sum[(i-1)*k]
            # Update the maximum and minimum weights encountered for this specific k
            current_max_W = max(current_max_W, W_i)
            current_min_W = min(current_min_W, W_i)

        # The difference for this k is max_weight - min_weight.
        # Update the overall maximum difference found across all k considered so far.
        max_overall_diff = max(max_overall_diff, current_max_W - current_min_W)

    # Print the final answer for this test case. Use sys.stdout.write for potentially faster output.
    sys.stdout.write(str(max_overall_diff) + "
")


# Read the number of test cases
T = int(sys.stdin.readline())
# Process each test case by calling the solve function
for _ in range(T):
    solve()