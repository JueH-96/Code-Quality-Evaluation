import sys

def main():
    N = int(sys.stdin.readline())
    
    # hourly_counts[h] will store the total number of employees
    # who can participate if the meeting starts at UTC hour h.
    hourly_counts = [0] * 24 

    for _ in range(N):
        W, X = map(int, sys.stdin.readline().split())
        
        # For this base, employees can participate if their local meeting start time S_local
        # is between 9:00 and 17:00, inclusive.
        # The meeting is one hour long. So, local start S_local means meeting is [S_local, S_local+1).
        # This interval must be within [9:00, 18:00).
        # Thus, S_local >= 9 and S_local+1 <= 18, which means 9 <= S_local <= 17.
        
        # We need to find the UTC start hour (S_utc) corresponding to a valid S_local.
        # S_local = (S_utc + X) % 24
        # So, S_utc = (S_local - X) % 24.
        # Python's % operator handles negative results in a way consistent with mathematical modulo:
        # e.g., (9 - 18) % 24 is -9 % 24, which evaluates to 15.
        
        for local_valid_start_hour in range(9, 18): # Iterates S_local from 9 to 17
            utc_start_hour_for_this_base = (local_valid_start_hour - X) % 24
            hourly_counts[utc_start_hour_for_this_base] += W
            
    max_participants = 0
    # The hourly_counts list will not be empty (it has 24 elements).
    # If all W_i are 0, all counts will be 0, and max will be 0.
    # Constraints state 1 <= N, so N is not 0.
    if hourly_counts: 
        max_participants = max(hourly_counts)
    
    print(max_participants)

if __name__ == '__main__':
    main()