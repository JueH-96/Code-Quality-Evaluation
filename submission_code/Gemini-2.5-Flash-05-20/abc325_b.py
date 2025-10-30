import sys

def solve():
    # Read N, the number of bases
    N = int(sys.stdin.readline())
    
    # Store base information (employees W_i, UTC offset X_i)
    # Each base is stored as a tuple (W_i, X_i)
    bases = []
    for _ in range(N):
        W, X = map(int, sys.stdin.readline().split())
        bases.append((W, X))
        
    # Initialize the maximum number of participating employees found so far.
    # This will store the global maximum across all possible meeting times.
    max_employees = 0
    
    # Iterate through all possible UTC start times for the one-hour meeting.
    # A meeting can start at any integer hour from 0 (0:00 UTC) to 23 (23:00 UTC).
    # Since the meeting is one hour long, a start time of 23:00 means it ends at 0:00.
    for utc_start_time in range(24):
        # Calculate the total employees for the current UTC start time.
        # This sum is reset for each potential UTC start time.
        current_employees = 0
        
        # For the current UTC_start_time, iterate through all bases
        # to determine how many employees can participate.
        for W_i, X_i in bases:
            # Calculate the local start time for this specific base.
            # The local time is (UTC_time + offset_X_i).
            # The modulo 24 (%) handles time wrapping around midnight (e.g., 23 + 3 = 26 becomes 2).
            local_start_time = (utc_start_time + X_i) % 24
            
            # An employee can participate only if the meeting time (one hour duration)
            # is COMPLETELY within their base's 9:00-18:00 local time slot.
            # This means:
            # 1. The meeting must start at or after 9:00 local time.
            # 2. The meeting must end at or before 18:00 local time.
            # Since it's a one-hour meeting, if it starts at S, it ends at S+1.
            # So, S >= 9 AND S+1 <= 18.
            # This simplifies to S >= 9 AND S <= 17.
            # Therefore, the valid local_start_time must be in the range [9, 17] inclusive.
            # In Python, this is `9 <= local_start_time < 18`.
            if 9 <= local_start_time < 18:
                current_employees += W_i
        
        # After checking all bases for the current `utc_start_time`,
        # update the overall maximum if this time yields more participants.
        max_employees = max(max_employees, current_employees)
        
    # Print the final maximum number of employees that can participate.
    print(max_employees)

# Call the solve function to execute the program logic.
if __name__ == '__main__':
    solve()