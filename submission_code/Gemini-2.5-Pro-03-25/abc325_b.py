import sys

# Function to solve the problem
def solve():
    # Read the number of bases
    N = int(sys.stdin.readline())
    
    # Array to store the total number of participants for each possible UTC start hour (0 to 23)
    # counts[T] will store the total employees who can attend if the meeting starts at T UTC.
    counts = [0] * 24
    
    # Iterate through each base, reading input line by line
    for _ in range(N):
        # Read the number of employees W_i and the local time offset X_i for base i
        W_i, X_i = map(int, sys.stdin.readline().split())
        
        # Determine the valid UTC start times for employees at this base to participate.
        # An employee can participate if the meeting (1 hour duration) is completely within 
        # the 9:00 to 18:00 time slot in their local time.
        # Let T be the start time in UTC. The meeting runs from T to T+1 UTC.
        # In local time at base i, the meeting starts at S_i = (T + X_i) % 24.
        # The meeting interval in local time is [S_i, S_i + 1).
        # This interval must be completely within [9, 18).
        # This requires S_i >= 9 and S_i + 1 <= 18.
        # Combining these conditions, we need 9 <= S_i <= 17.
        
        # Iterate through possible local start hours k that satisfy the condition (9 to 17 inclusive)
        for k in range(9, 18):
            # Calculate the corresponding UTC start time T for a local start time k.
            # We have k = (T + X_i) % 24.
            # Rearranging gives T = (k - X_i) % 24.
            # Python's % operator handles negative results correctly for modulo calculation in the range [0, 23].
            T = (k - X_i) % 24
            
            # Add the number of employees W_i to the total count for this UTC start time T.
            counts[T] += W_i
            
    # Find the maximum number of employees that can participate across all possible UTC start times (0 to 23).
    # Since N >= 1, the counts array will have at least one update if any W_i > 0.
    # If all W_i are 0, max will be 0 correctly.
    max_employees = max(counts)
    
    # Print the maximum number of employees
    print(max_employees)

# Call the solve function to execute the logic
solve()