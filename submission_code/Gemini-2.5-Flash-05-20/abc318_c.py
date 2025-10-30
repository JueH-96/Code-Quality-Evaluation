import sys

def solve():
    # Read N, D, P from the first line of input
    # Using sys.stdin.readline for potentially faster input in competitive programming
    N, D, P = map(int, sys.stdin.readline().split())

    # Read the list of fares F_i from the second line of input
    F = list(map(int, sys.stdin.readline().split()))

    # Sort fares in descending order.
    # This is a greedy approach: if we decide to use passes, we should always use them
    # for the days with the highest regular fares, as this maximizes the 'savings'.
    F.sort(reverse=True)

    # Calculate the sum of all regular fares.
    # This value represents the cost if we choose to pay the regular fare for every day,
    # which is our initial candidate for the minimum total cost (equivalent to using 0 passes).
    total_regular_fare_sum = sum(F)
    min_total_cost = total_regular_fare_sum

    # current_fares_covered_sum will accumulate the sum of fares for the days
    # that we decide to cover with one-day passes in the current iteration.
    current_fares_covered_sum = 0

    # Iterate through the sorted fares.
    # The loop variable 'i' represents the current index in the sorted F list.
    # In each iteration, we consider covering F[0]...F[i] with passes.
    # This means 'k = i + 1' days are covered by passes.
    for i in range(N):
        # Add the fare for the current day (which is the (i+1)-th highest fare)
        # to the sum of fares that will be covered by passes.
        current_fares_covered_sum += F[i]

        # 'k' is the total number of days we are covering with passes in this iteration.
        k = i + 1

        # Calculate the number of batches of passes needed for 'k' passes.
        # Since passes come in batches of D, we need ceil(k / D) batches.
        # The expression (k + D - 1) // D is a common integer division trick
        # to compute ceil(k / D) for positive integers k and D.
        num_batches = (k + D - 1) // D

        # Calculate the total cost of purchasing these batches of passes.
        cost_passes = num_batches * P

        # Calculate the sum of regular fares for the days that are NOT covered by passes.
        # This is the total sum of all fares minus the sum of fares for days
        # that are covered by passes in this configuration.
        cost_regular_fares = total_regular_fare_sum - current_fares_covered_sum

        # Calculate the total cost for the current configuration (cost of passes + cost of remaining regular fares).
        current_total_cost = cost_passes + cost_regular_fares

        # Update the overall minimum total cost found so far.
        min_total_cost = min(min_total_cost, current_total_cost)

    # Print the final minimum possible total cost.
    print(min_total_cost)

# Call the solve function to execute the program.
solve()