# NUMBER OF TEST CASES
N = int(input())

# LIST TO STORE THE TIME INTERVALS
intervals = []

# READ THE INPUTS
for i in range(N):
    T, D = map(int, input().split())
    intervals.append((T, D))

# SORT THE INTERVALS BASED ON THE START TIME
intervals.sort()

# LIST TO STORE THE END TIMES OF THE INTERVALS
end_times = []

# VARIABLE TO KEEP TRACK OF THE CURRENT TIME
current_time = 0

# VARIABLE TO KEEP TRACK OF THE MAXIMUM NUMBER OF PRINTS
max_prints = 0

# VARIABLE TO KEEP TRACK OF THE NEXT AVAILABLE TIME FOR PRINTING
next_available_time = 0

# ITERATE THROUGH THE INTERVALS
for T, D in intervals:
    # IF THE CURRENT TIME IS LESS THAN THE NEXT AVAILABLE TIME, ADVANCE THE CURRENT TIME
    if current_time < next_available_time:
        current_time = next_available_time
    
    # IF THE CURRENT TIME IS LESS THAN THE START TIME OF THE INTERVAL, ADVANCE THE CURRENT TIME
    if current_time < T:
        current_time = T
    
    # IF THE CURRENT TIME IS LESS THAN THE END TIME OF THE INTERVAL, ADVANCE THE CURRENT TIME
    if current_time < T + D:
        current_time = T + D
    
    # IF THE CURRENT TIME IS LESS THAN THE NEXT AVAILABLE TIME, ADVANCE THE CURRENT TIME
    if current_time < next_available_time:
        current_time = next_available_time
    
    # INCREMENT THE MAXIMUM NUMBER OF PRINTS
    max_prints += 1
    
    # UPDATE THE NEXT AVAILABLE TIME
    next_available_time = current_time + 1

# PRINT THE MAXIMUM NUMBER OF PRINTS
print(max_prints)