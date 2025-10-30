M = int(input())
days = list(map(int, input().split()))

# Check if the total number of days is odd
if sum(days) % 2 == 0:
    print("Error: The total number of days is even.")
else:
    # Find the middle day
    middle_day = (sum(days) + 1) // 2
    for i in range(1, M + 1):
        if sum(days[:i]) >= middle_day:
            print(i, middle_day - sum(days[:i-1]))
            break