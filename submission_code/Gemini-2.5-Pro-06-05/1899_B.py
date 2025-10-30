import sys
import math

def solve():
    """
    Solves a single test case by reading input, processing, and printing the result.
    """
    # Read input for a single test case
    try:
        n = int(sys.stdin.readline())
        a = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Handle potential empty lines or malformed input at the end of a file
        return

    # 1. Compute prefix sums for efficient range sum queries.
    # Python's arbitrary-precision integers handle large sums, so no overflow.
    prefix_sum = [0] * (n + 1)
    for i in range(n):
        prefix_sum[i + 1] = prefix_sum[i] + a[i]

    # 2. Find all divisors of n. These are the possible values for k.
    divisors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i * i != n:
                divisors.append(n // i)
    
    # 3. Iterate through divisors to find the one maximizing the weight difference.
    max_diff = 0
    for k in divisors:
        # The case k=n results in only one truck. The difference is 0.
        # Since max_diff is initialized to 0, we can skip this case.
        if k == n:
            continue

        min_w = float('inf')
        max_w = float('-inf')
        
        num_trucks = n // k
        for j in range(num_trucks):
            # The weight of the truck for the j-th group of k boxes
            # is calculated efficiently using the prefix sum array.
            weight = prefix_sum[(j + 1) * k] - prefix_sum[j * k]
            min_w = min(min_w, weight)
            max_w = max(max_w, weight)
        
        current_diff = max_w - min_w
        max_diff = max(max_diff, current_diff)

    print(max_diff)

def main():
    """
    Main function to handle multiple test cases as per the problem format.
    """
    try:
        num_test_cases = int(sys.stdin.readline())
        for _ in range(num_test_cases):
            solve()
    except (IOError, ValueError):
        # Gracefully handle end of file or malformed input for the number of test cases.
        pass

if __name__ == "__main__":
    main()