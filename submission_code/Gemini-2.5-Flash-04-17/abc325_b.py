import sys

def solve():
    # Read the number of bases
    N = int(sys.stdin.readline())

    # Store the number of employees (W_i) and local offsets (X_i) for each base
    bases = []
    for _ in range(N):
        W, X = map(int, sys.stdin.readline().split())
        bases.append((W, X))

    # Initialize the maximum number of participating employees found so far
    # The maximum possible number of employees is N * max(W_i) which can be up to 1000 * 10^6 = 10^9.
    # Python's built-in int type handles arbitrary size integers, so no overflow concern here.
    max_employees = 0

    # Iterate through all possible integer UTC start times for the meeting.
    # The meeting is a one-hour slot. Let the UTC start time be t (an integer hour, e.g., 14 means 14:00 UTC).
    # The meeting is from t:00 UTC to (t+1):00 UTC.
    # The local time at base i when it is 0:00 UTC is X_i:00.
    # This means the local time at base i when it is t:00 UTC is (t + X_i) mod 24 hours.
    # Since the local time pattern repeats every 24 hours of UTC time,
    # we only need to check integer UTC start times t from 0 to 23.
    # This covers all 24 distinct possibilities for the integer local start hour
    # (t + X_i) % 24 relative to any fixed offset X_i.
    for t in range(24):
        # Calculate the total number of participating employees for the current UTC start time t
        current_employees = 0
        for W, X in bases:
            # Calculate the local time (hour) at base i when the meeting starts (at t:00 UTC)
            # The local time = (UTC hour + offset) modulo 24 hours.
            # X_i is the local hour when UTC is 0. So, local time = (t + X) % 24.
            local_start_hour = (t + X) % 24

            # Check if the one-hour meeting starting at local_start_hour:00 local time
            # is completely within the 9:00-18:00 local time slot.
            # The 9:00-18:00 slot means from 9:00 up to (but not including) 18:00.
            # A one-hour meeting starting at local time L:00 is from L:00 to L+1:00.
            # For this interval [L:00, L+1:00) to be completely within [9:00, 18:00),
            # the start time L must be >= 9:00 and the end time L+1 must be <= 18:00.
            # Since L is an integer hour, this means 9 <= L and L + 1 <= 18.
            # The second condition simplifies to L <= 17.
            # So, the condition for the local start hour is 9 <= L <= 17.
            if 9 <= local_start_hour <= 17:
                # If the local start hour is within the valid window [9, 17],
                # all employees at this base can participate in the meeting.
                current_employees += W

        # After checking all bases for the current UTC start time t,
        # update the maximum number of participating employees found so far.
        max_employees = max(max_employees, current_employees)

    # The maximum value found over all tested integer UTC start times (0-23) is the answer.
    print(max_employees)

# Execute the solve function to run the program
solve()