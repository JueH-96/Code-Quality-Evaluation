# YOUR CODE HERE
H = int(input())

# 'day' will store the current day number.
# 'plant_height_morning' will store the plant's height in the morning of 'day'.
# The height in the morning of day 'k' is the sum of growth from night 0
# up to night (k-1), which is 2^0 + 2^1 + ... + 2^(k-1) = 2^k - 1.

day = 0

# We iterate, incrementing the day, and calculating the plant's height
# for that morning, until it exceeds Takahashi's height H.
while True:
    day += 1
    
    # Calculate the plant's height in the morning of the current 'day'.
    # This is 2^day - 1.
    # (1 << day) is an efficient way to compute 2 raised to the power of 'day'.
    plant_height_morning = (1 << day) - 1
    
    # Check if the plant's height is strictly greater than Takahashi's height H.
    if plant_height_morning > H:
        # If it is, this is the first such day. Print it and exit the loop.
        print(day)
        break