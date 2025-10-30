# YOUR CODE HERE
N, M = map(int, input().split())
S = input().strip()

# Split the schedule into periods between washing days
periods = []
current_period = []

for day in S:
    if day == '0':
        if current_period:
            periods.append(current_period)
            current_period = []
    else:
        current_period.append(day)

# Don't forget the last period if it doesn't end with '0'
if current_period:
    periods.append(current_period)

# For each period, calculate the minimum logo T-shirts needed
max_logo_needed = 0

for period in periods:
    # Count days requiring T-shirts
    meal_days = period.count('1')
    event_days = period.count('2')
    total_days = meal_days + event_days
    
    # We need at least event_days logo T-shirts
    # We can use plain T-shirts for meal days if available
    # But if total_days > M (plain shirts), we need extra logo shirts
    logo_needed = event_days
    if total_days > M:
        # We need more T-shirts than we have plain ones
        # So we need additional logo T-shirts
        logo_needed = total_days - M
    
    # Make sure we have at least event_days logo shirts
    logo_needed = max(logo_needed, event_days)
    
    max_logo_needed = max(max_logo_needed, logo_needed)

print(max_logo_needed)