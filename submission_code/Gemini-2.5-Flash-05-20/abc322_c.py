import sys

def solve():
    # Read N and M from the first line
    N, M = map(int, sys.stdin.readline().split())

    # Read the list of days A where fireworks are launched
    A = list(map(int, sys.stdin.readline().split()))

    # Create a boolean array to quickly check if a day has fireworks.
    # The size is N+1 to allow 1-based indexing, matching day numbers.
    # is_firework_day[i] will be True if fireworks are on day i, False otherwise.
    is_firework_day = [False] * (N + 1)
    for day in A:
        is_firework_day[day] = True

    # Initialize an array to store the results for each day.
    # ans[i] will store the number of days until the next firework launch from day i.
    # Size N+1 for 1-based indexing.
    ans = [0] * (N + 1)

    # 'next_firework_day' keeps track of the most recent firework day
    # encountered as we iterate backwards from N down to 1.
    # Since A_M = N is guaranteed, day N always has fireworks, so we can
    # initialize next_firework_day to N.
    next_firework_day = N 

    # Iterate from the last day (N) down to the first day (1)
    for i in range(N, 0, -1):
        if is_firework_day[i]:
            # If day 'i' has fireworks, the answer for day 'i' is 0.
            # Also, this day 'i' becomes the new nearest future firework day
            # for any days before it.
            ans[i] = 0
            next_firework_day = i
        else:
            # If day 'i' does not have fireworks, the answer is the difference
            # between the 'next_firework_day' (which is the first firework day
            # on or after 'i') and 'i'.
            ans[i] = next_firework_day - i

    # Print the results for each day from 1 to N, each on a new line.
    # Using sys.stdout.write for potentially faster output.
    output_buffer = []
    for i in range(1, N + 1):
        output_buffer.append(str(ans[i]))
    sys.stdout.write("
".join(output_buffer) + "
")

# Call the solve function to run the program
solve()