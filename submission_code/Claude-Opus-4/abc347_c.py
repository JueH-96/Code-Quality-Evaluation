# YOUR CODE HERE
N, A, B = map(int, input().split())
D = list(map(int, input().split()))

week_length = A + B

# For each possible starting day (0 to week_length-1)
for start in range(week_length):
    all_holidays = True
    
    # Check if all plans fall on holidays
    for d in D:
        # Calculate which day of the week this plan falls on
        day_of_week = (start + d) % week_length
        
        # Check if it's a holiday (days 0 to A-1 in 0-indexed)
        if day_of_week >= A:
            all_holidays = False
            break
    
    if all_holidays:
        print("Yes")
        exit()

print("No")