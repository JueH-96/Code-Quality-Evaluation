import sys

def solve():
    # Read N, T, P from the first line of input
    # N: number of people
    # T: target hair length
    # P: target number of people
    N, T, P = map(int, sys.stdin.readline().split())

    # Read the list L containing current hair lengths for N people
    L = list(map(int, sys.stdin.readline().split()))

    # Create a list to store the number of days each person needs for their hair
    # to reach length T or more.
    days_needed_for_each_person = []

    # Iterate through each person's current hair length
    for current_L in L:
        # Calculate the days needed for this specific person:
        # If current_L is already greater than or equal to T, this person
        # effectively needs 0 more days to meet the condition.
        # Otherwise, they need T - current_L days for their hair to grow to T.
        days = max(0, T - current_L)
        days_needed_for_each_person.append(days)

    # Sort the list of 'days_needed_for_each_person' in ascending order.
    # This ordering is crucial:
    # The first element will be the minimum days needed by anyone to reach T.
    # The second element will be the minimum days needed by the second person, and so on.
    days_needed_for_each_person.sort()

    # The problem asks for the minimum number of days after which P or more people
    # have hair length at least T.
    # After sorting, the (P-1)-th element (using 0-based indexing) in the
    # 'days_needed_for_each_person' list represents the number of days required
    # for at least P people to satisfy the condition.
    # For instance, if P=1, we need the minimum days for 1 person, which is the 0th element.
    # If P=N, we need the maximum days among all persons, which is the (N-1)th element.
    result_days = days_needed_for_each_person[P - 1]

    # Print the calculated result to standard output
    print(result_days)

# This check ensures that the solve() function is called only when the script is
# executed directly (not when imported as a module).
if __name__ == '__main__':
    solve()