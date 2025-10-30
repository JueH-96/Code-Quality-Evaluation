# Read the number of medicines and the threshold K
N, K = map(int, input().split())

# Initialize the day counter
day = 1

# Loop through each medicine
for _ in range(N):
    a, b = map(int, input().split())
    # If the current day is within the prescription period, subtract the pills from K
    if day <= a:
        K -= b
        # If K is less than or equal to 0, we found the day where Takahashi takes K pills or less
        if K <= 0:
            break
    else:
        # If the current day is past the prescription period, move to the next day
        day = a + 1
        K = K + b - 1  # Adjust K for the new day

# Output the day
print(day)