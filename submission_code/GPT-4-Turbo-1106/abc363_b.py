# Read the inputs from stdin
N, T, P = map(int, input().split())
L = list(map(int, input().split()))

# Initialize the number of days to 0
days = 0

# Check if the condition is already satisfied
if len([l for l in L if l >= T]) >= P:
    print(0)
else:
    # Iterate until the condition is satisfied
    while True:
        days += 1
        # Count the number of people with hair length at least T
        count = len([l for l in L if l + days >= T])
        # If the count is at least P, break the loop
        if count >= P:
            break
    # Print the number of days
    print(days)