from typing import List

def max_employees_meeting(N: int, W_i: List[int], X_i: List[int]) -> int:
    # Sort the bases by their start time (X_i)
    bases = sorted(zip(X_i, W_i, range(1, N+1)))
    
    # Initialize the meeting time range
    start_time = 9
    end_time = 18
    
    # Initialize the total number of participating employees
    total_employees = 0
    
    # Iterate through the bases
    for x, w, i in bases:
        # Calculate the start and end times for the base
        base_start = (x + 9) % 24
        base_end = (x + 18) % 24
        
        # Check if the meeting time is within the base's 9:00-18:00 time slot
        if start_time >= base_start and end_time <= base_end:
            total_employees += w
    
    return total_employees

# Read the input
N = int(input())
W_i = []
X_i = []
for _ in range(N):
    w, x = map(int, input().split())
    W_i.append(w)
    X_i.append(x)

# Solve the problem and print the result
print(max_employees_meeting(N, W_i, X_i))