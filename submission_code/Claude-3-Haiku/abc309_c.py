# Read the input
N, K = map(int, input().split())
medicines = []
for _ in range(N):
    a, b = map(int, input().split())
    medicines.append((a, b))

# Sort the medicines by the number of days
medicines.sort()

# Initialize the total number of pills to 0 and the current day to 1
total_pills = 0
current_day = 1

# Iterate through the medicines and find the first day when the total number of pills is less than or equal to K
for days, pills in medicines:
    total_pills += pills
    if total_pills <= K:
        print(current_day)
        break
    current_day += days
else:
    print(current_day)