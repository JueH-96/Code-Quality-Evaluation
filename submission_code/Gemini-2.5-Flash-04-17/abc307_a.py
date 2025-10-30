import sys

# Read the number of weeks, N.
# N is an integer on the first line of standard input.
N = int(sys.stdin.readline())

# Read the 7*N daily step counts.
# These are space-separated integers on the second line of standard input.
# sys.stdin.readline() reads the entire line as a string.
# split() splits the string into a list of substrings based on whitespace.
# map(int, ...) applies the int() function to each substring in the list, converting them to integers.
# list(...) converts the result of the map object into a list of integers.
steps = list(map(int, sys.stdin.readline().split()))

# Initialize an empty list to store the sum of steps for each week.
weekly_sums = []

# Iterate through each week. There are N weeks in total.
# The loop variable 'i' represents the 0-based index of the week (0 for the first week, 1 for the second, ..., N-1 for the N-th week).
for i in range(N):
    # Calculate the starting index for the current week in the 'steps' list.
    # The first week (i=0) corresponds to the first 7 elements (indices 0 to 6).
    # The second week (i=1) corresponds to the next 7 elements (indices 7 to 13).
    # The i-th week (0-indexed) corresponds to the 7 elements starting at index i * 7.
    start_index = i * 7

    # Calculate the ending index for the current week (exclusive in Python list slicing).
    # The slice should include 7 days starting from start_index.
    # So, the indices included are start_index, start_index+1, ..., start_index+6.
    # The ending index for the slice is start_index + 7.
    end_index = start_index + 7

    # Calculate the sum of steps for the current week.
    # steps[start_index : end_index] creates a slice (a portion) of the 'steps' list
    # containing the step counts for the 7 days of the current week.
    # sum() calculates the total of the elements in this slice.
    current_week_sum = sum(steps[start_index : end_index])

    # Add the calculated weekly sum to our result list.
    weekly_sums.append(current_week_sum)

# Print the list of weekly sums.
# The problem requires the output to be space-separated integers on a single line.
# The '*' operator unpacks the 'weekly_sums' list into individual arguments for the print function.
# By default, the print function separates its arguments with spaces and adds a newline character at the end.
print(*weekly_sums)