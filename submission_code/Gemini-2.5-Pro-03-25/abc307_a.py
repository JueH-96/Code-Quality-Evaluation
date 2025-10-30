# YOUR CODE HERE
import sys

def solve():
    """
    Reads the number of weeks N and the daily step counts for 7N days.
    Calculates the total steps for each week and prints them space-separated.
    """
    # Read the number of weeks, N, from the first line of standard input.
    n = int(sys.stdin.readline())

    # Read the list of daily step counts, A, from the second line of standard input.
    # The input guarantees exactly 7*N integers are provided, separated by spaces.
    # map(int, ...) converts each space-separated string item to an integer.
    # list(...) converts the map object into a list.
    a = list(map(int, sys.stdin.readline().split()))

    # Initialize an empty list to store the total steps calculated for each week.
    weekly_sums = []

    # Iterate through each week. Since there are N weeks, the loop runs N times.
    # We use 'i' as the week index (0-based, representing week 1, 2, ..., N).
    for i in range(n):
        # Calculate the starting index in the list 'a' for the current week 'i'.
        # Week 0 (the first week) starts at index 0.
        # Week 1 (the second week) starts at index 7.
        # Week 'i' (the (i+1)-th week) starts at index i * 7.
        start_index = i * 7

        # Calculate the ending index (exclusive) for the slice representing the current week's steps.
        # Each week consists of 7 days. The steps for week 'i' are at indices
        # start_index, start_index + 1, ..., start_index + 6.
        # In Python slicing a[start:end], the slice includes elements from 'start' up to,
        # but not including, 'end'. So, the 'end' index should be start_index + 7.
        end_index = start_index + 7

        # Calculate the sum of steps for the current week.
        # a[start_index:end_index] creates a sublist (slice) containing the 7 daily step counts for the current week.
        # The built-in function sum() efficiently computes the total of the elements in this sublist.
        current_week_sum = sum(a[start_index:end_index])

        # Append the calculated sum for the current week to our list 'weekly_sums'.
        weekly_sums.append(current_week_sum)

    # After the loop finishes, the 'weekly_sums' list contains N elements,
    # where the k-th element (0-based index k) is the total steps for week k+1.
    # We need to print these sums separated by spaces.
    # The '*' operator before 'weekly_sums' unpacks the list.
    # For example, if weekly_sums is [28000, 35000], then print(*weekly_sums)
    # is equivalent to print(28000, 35000).
    # The print function, by default, separates its arguments with a single space.
    # This matches the required output format.
    print(*weekly_sums)

# Execute the main logic of the program by calling the solve function.
solve()