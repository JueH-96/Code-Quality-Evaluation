# YOUR CODE HERE
import sys

# Function to solve a single test case
def solve():
    # Read the number of boxes
    n = int(sys.stdin.readline())
    # Read the weights of the boxes
    # Use map with int to convert split strings to integers
    a = list(map(int, sys.stdin.readline().split()))

    # Calculate prefix sums
    # P[i] will store the sum of the first i elements of a (a[0] to a[i-1])
    # P is 1-indexed conceptually for sums, but 0-indexed for list access
    # P[0] = sum of first 0 elements = 0
    # P[i] = sum of first i elements (a[0]...a[i-1])
    P = [0] * (n + 1)
    for i in range(n):
        P[i+1] = P[i] + a[i]

    # Initialize the maximum difference found so far
    # If there is only one truck (m=1), the difference is 0.
    max_diff = 0

    # Iterate through possible number of trucks m
    # According to the problem, trucks must hold the same number of boxes, k.
    # This means n must be divisible by k. The number of trucks will be m = n / k.
    # So, m must be a divisor of n.
    # We iterate through all possible integer values of m from 1 to n.
    # If m is a divisor of n, then k = n // m is a valid number of boxes per truck.
    for m in range(1, n + 1):
        if n % m == 0:
            # This m is a valid number of trucks.
            k = n // m # Number of boxes per truck for this configuration
            
            # Calculate the total weight for each of the m trucks
            truck_weights = []
            # Truck j (0-indexed, j goes from 0 to m-1) contains boxes from index j*k to (j+1)*k - 1
            # The sum of elements a[start:end] (exclusive end index) is P[end] - P[start]
            # using 0-indexed array a and prefix sums P where P[i] = sum(a[0:i]).
            for j in range(m):
                # The indices in the original list 'a' for the j-th truck are from j*k to (j+1)*k - 1
                # Using prefix sums P: the sum is P[(j+1)*k] - P[j*k]
                weight = P[(j+1)*k] - P[j*k]
                truck_weights.append(weight)

            # If there is more than one truck (m > 1), calculate the difference
            # If m == 1, there is only one truck, and the difference is 0.
            # max_diff is initialized to 0, so the m=1 case is handled correctly without explicit check.
            # The problem asks for the maximum absolute difference between *two* trucks.
            # If m=1, there are no *two* trucks, so the difference is effectively 0.
            # The min() and max() operations require m > 0 elements. m starts from 1, so this is fine.
            # If m > 1, calculate the range (max - min)
            if m > 1:
                min_w = min(truck_weights)
                max_w = max(truck_weights)
                # Update the maximum difference found so far across all valid configurations
                max_diff = max(max_diff, max_w - min_w)

    # Print the result for this test case
    print(max_diff)

# Read the total number of test cases
t = int(sys.stdin.readline())

# Process each test case by calling the solve function
for _ in range(t):
    solve()