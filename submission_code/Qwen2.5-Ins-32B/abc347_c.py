import sys

def is_possible_holiday(N, A, B, D):
    # Calculate the maximum number of days that can be a holiday
    max_holiday_day = A + (B * ((max(D) - A) // B))
    
    # Check if all days are within the holiday range
    for day in D:
        if day > max_holiday_day:
            return "No"
    return "Yes"

# Read input from stdin
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = int(data[1])
B = int(data[2])
D = list(map(int, data[3:]))

# Solve the problem and write the answer to stdout
print(is_possible_holiday(N, A, B, D))