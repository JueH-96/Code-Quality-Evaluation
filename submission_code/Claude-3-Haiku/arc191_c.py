import math

def find_pair(N):
    # Initialize a to 2 and m to 3 (the smallest possible values)
    a, m = 2, 3
    
    # Iterate through possible values of a and m
    while True:
        # Check if a^N - 1 is divisible by m
        if pow(a, N, m) == 1:
            return a, m
        
        # Increment a and m
        a += 1
        if a > m:
            m += 1
            a = 2

# Read the number of test cases
T = int(input())

# Solve each test case
for _ in range(T):
    N = int(input())
    a, m = find_pair(N)
    print(a, m)