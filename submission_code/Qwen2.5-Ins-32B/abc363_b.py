import sys

def solve(N, T, P, L):
    # Count the number of people whose hair length is already at least T
    count = sum(1 for x in L if x >= T)
    
    # If the condition is already satisfied, return 0
    if count >= P:
        return 0
    
    # Calculate the number of days needed for P people to have hair length at least T
    days = 0
    while count < P:
        days += 1
        count += sum(1 for x in L if x + days == T)
    
    return days

# Read input from stdin
N, T, P = map(int, input().split())
L = list(map(int, input().split()))

# Solve the problem and print the answer to stdout
print(solve(N, T, P, L))