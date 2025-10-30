# YOUR CODE HERE
import sys

def solve():
    # Read N, T, P from the first line of standard input
    n, t, p = map(int, sys.stdin.readline().split())
    # Read the list of hair lengths L from the second line of standard input
    l = list(map(int, sys.stdin.readline().split()))

    # Count the number of people whose hair length is already at least T
    initial_count = 0
    # Store the number of days needed for people whose hair length is currently less than T
    # to reach length T.
    days_needed = []

    # Iterate through each person's initial hair length
    for length in l:
        if length >= t:
            # If hair length is already >= T, increment the initial count
            initial_count += 1
        else:
            # If hair length is < T, calculate the days needed to reach T
            # Hair grows by 1 per day. To reach T from length, need T - length days.
            needed = t - length
            days_needed.append(needed)

    # Check if the required number of people (P) already have hair length >= T initially (at day 0)
    if initial_count >= p:
        # If the condition is already satisfied, print 0 days.
        print(0)
        return # Exit the function

    # If the condition is not met initially, we need to wait for some days.
    # We need 'k' more people to reach the threshold T.
    k = p - initial_count

    # The 'days_needed' list contains the number of days each person (initially below T)
    # needs to reach the threshold T.
    # To find the first day when at least P people have length >= T, we need to find
    # the time it takes for the k-th fastest person (among those initially below T)
    # to reach the threshold.

    # Sort the 'days_needed' list in ascending order. The k-th smallest value
    # will give us the minimum number of days required.
    days_needed.sort()

    # Before accessing days_needed[k-1], ensure k is valid.
    # Since initial_count < p, we know k = p - initial_count >= 1.
    # The number of people initially below T is len(days_needed) = n - initial_count.
    # Since p <= n, we have p - initial_count <= n - initial_count.
    # So, k <= len(days_needed).
    # This guarantees that 1 <= k <= len(days_needed), meaning the index k-1 is valid
    # (ranging from 0 to len(days_needed)-1).

    # The result is the k-th smallest number of days needed, which is at index k-1
    # in the sorted list (because lists are 0-indexed).
    result = days_needed[k-1]

    # Print the calculated minimum number of days.
    print(result)

# Call the solve function to execute the logic
solve()