# YOUR CODE HERE
def find_middle_day():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    M = int(data[0])
    D = list(map(int, data[1:]))
    
    # Calculate the total number of days in the year
    total_days = sum(D)
    
    # Calculate the middle day index (1-based)
    middle_day_index = (total_days + 1) // 2
    
    # Find the month and day corresponding to the middle day
    current_day_count = 0
    for month in range(M):
        if current_day_count + D[month] >= middle_day_index:
            day_in_month = middle_day_index - current_day_count
            print(month + 1, day_in_month)
            return
        current_day_count += D[month]

find_middle_day()