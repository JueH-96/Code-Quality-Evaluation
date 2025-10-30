# YOUR CODE HERE
N, K = map(int, input().split())
medicines = []
for _ in range(N):
    a, b = map(int, input().split())
    medicines.append((a, b))

# Calculate total pills on day 1
total_pills = sum(b for a, b in medicines)

# If already <= K on day 1
if total_pills <= K:
    print(1)
else:
    # Sort medicines by expiration day
    medicines.sort()
    
    # Check each day when a medicine expires
    for i in range(N):
        day = medicines[i][0] + 1  # Day after medicine i expires
        
        # Calculate pills on this day
        pills_on_day = 0
        for a, b in medicines:
            if day <= a:
                pills_on_day += b
        
        if pills_on_day <= K:
            print(day)
            break