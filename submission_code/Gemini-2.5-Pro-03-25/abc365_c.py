# YOUR CODE HERE
import sys
import bisect

# Function implementing the core logic
def solve():
    # Read N (number of people) and M (budget) from input
    # Use sys.stdin.readline for potentially faster input reading
    N, M = map(int, sys.stdin.readline().split())
    # Read the list of transportation costs A
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate the total sum of all transportation costs
    # This represents the maximum possible total subsidy required if x is very large (x >= max(A_i))
    total_A = sum(A) # Python's sum() is efficient

    # Check if the total cost is already within budget.
    # If total_A <= M, then even if everyone receives their full transportation cost as subsidy,
    # the budget is sufficient. In this case, the subsidy limit x can be arbitrarily large.
    if total_A <= M:
        print("infinite")
        return

    # If total_A > M, the maximum subsidy limit x must be finite.
    # Sort the array A. This is crucial for efficiently calculating the total subsidy S(x) 
    # for any given x using prefix sums and binary search properties.
    A.sort()

    # Calculate prefix sums P. P[k] stores the sum of the first k elements (0-indexed) of the sorted array A.
    # Specifically, P[0] = 0, and P[k] = A[0] + A[1] + ... + A[k-1] for k > 0.
    # P will have N+1 elements.
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P[i] + A[i]

    # Perform binary search to find the maximum possible integer value of x.
    # We are looking for the largest integer x such that the total subsidy S(x) = sum(min(x, A_i)) <= M.
    
    # The lower bound L for x is 0, since x must be a non-negative integer.
    L = 0 
    
    # The upper bound R must be strictly greater than the maximum possible answer.
    # We established that if the answer is finite (i.e., total_A > M), 
    # the maximum possible value for x must be less than or equal to max(A_i).
    # Since the maximum value for any A_i is 10^9 according to the constraints,
    # an upper bound of R = 10**9 + 1 is sufficient and safe. This ensures R is strictly greater than any possible optimal x.
    R = 10**9 + 1 

    # The binary search loop continues as long as the search interval [L, R) has size greater than 1.
    # The goal is to shrink the interval [L, R) until R = L + 1.
    while R - L > 1:
        # Calculate the middle value `mid` as a candidate for x.
        # Use integer division // to ensure mid is an integer.
        mid = (L + R) // 2
        
        # Calculate the total subsidy S(mid) for the candidate limit `mid`.
        # S(mid) = sum(min(mid, A_i) for i in range(N)).
        
        # Find k, the count of people whose transportation cost A_i is less than or equal to mid.
        # We use `bisect.bisect_right(A, mid)` on the sorted array A.
        # It returns the index `k` such that all elements A[0...k-1] are <= mid, 
        # and all elements A[k...N-1] are > mid.
        # This index `k` is exactly the number of people with A_i <= mid.
        k = bisect.bisect_right(A, mid)
        
        # The total subsidy S(mid) can be calculated efficiently using the prefix sums P and index k:
        # - For the first k people (where A_i <= mid), the subsidy is A_i. The sum of their subsidies is P[k].
        # - For the remaining N-k people (where A_i > mid), the subsidy is capped at mid. The sum of their subsidies is (N-k) * mid.
        current_subsidy = P[k] + (N - k) * mid
        
        # Compare the calculated total subsidy with the available budget M.
        if current_subsidy <= M:
            # If the subsidy is within budget, `mid` is a feasible value for x.
            # Since we want the maximum possible x, `mid` or potentially a larger value could be the answer.
            # We update the lower bound L = mid to explore the range [mid, R).
            # This maintains the invariant that S(L) <= M.
            L = mid
        else:
            # If the subsidy exceeds the budget, `mid` is too large to be a valid limit x.
            # Since S(x) is a non-decreasing function, any value x >= mid will also result in a subsidy > M.
            # Therefore, we must search for smaller values by updating the upper bound R = mid, exploring the range [L, mid).
            # This maintains the invariant that S(R) > M (or R remains an upper bound where S(R) could be > M).
            R = mid

    # After the loop terminates, we have R = L + 1.
    # L represents the largest integer value tested for which the condition S(L) <= M holds.
    # R represents the smallest integer value tested (or initial upper bound) for which S(R) > M.
    # Therefore, L is the maximum possible integer value for the subsidy limit x.
    print(L)

# Execute the solve function to run the program
solve()