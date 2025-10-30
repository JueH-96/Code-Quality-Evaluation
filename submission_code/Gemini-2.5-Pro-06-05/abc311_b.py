# YOUR CODE HERE
import sys

def solve():
    """
    This function reads the input, solves the problem, and prints the output.
    """
    # Read the number of people (N) and the number of days (D)
    try:
        line = sys.stdin.readline()
        # Handle empty input at the end of file
        if not line:
            return
        N, D = map(int, line.split())
    except (IOError, ValueError):
        # This case is unlikely given the problem constraints but is good practice.
        return

    # Read the N schedules for the D days
    schedules = [sys.stdin.readline().strip() for _ in range(N)]

    # Step 1: Determine the collective availability for each day.
    # A day is considered 'available' if and only if ALL people are free ('o') on that day.
    # We will build a string representing this collective availability, e.g., "ooxox",
    # where 'o' denotes a day everyone is free, and 'x' denotes a day someone is occupied.
    
    collective_availability_chars = []
    for day_index in range(D):
        # We check if all people's schedules show 'o' for the current day.
        # The all() function with a generator expression is efficient as it short-circuits.
        is_day_available = all(s[day_index] == 'o' for s in schedules)
        
        if is_day_available:
            collective_availability_chars.append('o')
        else:
            collective_availability_chars.append('x')
            
    availability_string = "".join(collective_availability_chars)

    # Step 2: Find the longest consecutive run of 'o's (available days).
    # A Pythonic way to do this is to split the availability string by 'x'.
    # This partitions the string into substrings of consecutive 'o's.
    # For example, "ooxoxxooo" would split into ["oo", "o", "", "ooo"].
    streaks_of_available_days = availability_string.split('x')

    # Step 3: Find the length of the longest streak.
    # We use the max() function with the `key=len` argument to find the longest
    # substring in our list of streaks. Then we get its length.
    # If there are no 'o's at all, the list will contain only empty strings, and the
    # length of the longest one is correctly 0.
    
    # Since D >= 1, streaks_of_available_days will not be empty.
    max_len = len(max(streaks_of_available_days, key=len))

    # Print the final result to standard output.
    print(max_len)

# Run the solution
solve()