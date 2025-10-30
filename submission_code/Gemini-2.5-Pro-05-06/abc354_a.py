H = int(input())

plant_height = 0
ans_day = 0
# power_of_2 will take values 2^0, 2^1, 2^2, ...
# These are the amounts of growth on nights 0, 1, 2, ...
# (Night of day i means growth is 2^i)
power_of_2 = 1 

# The loop continues as long as the plant's height is not strictly greater than H.
# In each iteration, `ans_day` is incremented, representing the morning of that day.
# `plant_height` is updated to be the height on the morning of day `ans_day`.
while plant_height <= H:
    # Add the growth from the night that just finished.
    # If ans_day is currently k (meaning k days/mornings have passed),
    # this growth is from night k (0-indexed), which is 2^k.
    plant_height += power_of_2
    
    # Prepare the growth amount for the next night.
    power_of_2 *= 2
    
    # Increment the day counter.
    ans_day += 1

print(ans_day)