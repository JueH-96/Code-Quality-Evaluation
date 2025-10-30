def determine_winner(n):
    if n % 3 == 0:
        return "Second"
    else:
        return "First"

# Read the number of test cases
t = int(input().strip())

# Process each test case
for _ in range(t):
    n = int(input().strip())
    print(determine_winner(n))