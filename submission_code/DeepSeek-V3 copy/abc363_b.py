# Read input values
N, T, P = map(int, input().split())
L = list(map(int, input().split()))

# Check if the condition is already satisfied
count = sum(1 for l in L if l >= T)
if count >= P:
    print(0)
else:
    # Calculate the required days
    days = 0
    while True:
        days += 1
        count = sum(1 for l in L if l + days >= T)
        if count >= P:
            print(days)
            break