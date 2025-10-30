# Read input
H = int(input())

# Initialize variables
# day: Represents the current day number (starting from 0 for germination day).
#      The loop variable 'day' will represent the day whose morning height we are calculating.
# current_height: Represents the height of the plant in the morning of the current 'day'.
day = 0
current_height = 0

# Simulate day by day until the height in the morning is strictly greater than H.
# The loop iterates for day = 1, 2, 3, ...
while True:
    # Move to the next day.
    day += 1
    
    # Calculate the growth that happened during the night of the *previous* day (day-1).
    # The growth on night i (day i) is 2^i.
    # So, growth on night (day-1) is 2^(day-1).
    growth_previous_night = 2**(day - 1)
    
    # Add the growth from the night of the previous day to the height
    # measured in the morning of the previous day.
    # This sum gives the height in the morning of the current day ('day').
    # The height in the morning of day d is sum_{i=0}^{d-1} 2^i.
    current_height += growth_previous_night
    
    # Check if the plant's height in the morning of the current day ('day')
    # is strictly greater than Takahashi's height H.
    if current_height > H:
        # If it is, this is the first such day. Print the day number.
        print(day)
        # Exit the simulation loop.
        break