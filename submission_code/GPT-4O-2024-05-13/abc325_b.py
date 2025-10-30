# YOUR CODE HERE
def max_employees_participating(N, bases):
    max_employees = 0
    
    for utc_hour in range(24):
        current_employees = 0
        for W, X in bases:
            local_start = (utc_hour + X) % 24
            local_end = (utc_hour + X + 1) % 24
            if (9 <= local_start < 18) and (9 < local_end <= 18):
                current_employees += W
        max_employees = max(max_employees, current_employees)
    
    return max_employees

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
bases = []
for i in range(1, len(data), 2):
    W = int(data[i])
    X = int(data[i+1])
    bases.append((W, X))

# Calculate and print the result
print(max_employees_participating(N, bases))