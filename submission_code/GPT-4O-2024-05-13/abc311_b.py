# YOUR CODE HERE
def max_consecutive_free_days(N, D, schedules):
    max_days = 0
    current_streak = 0
    
    for day in range(D):
        all_free = True
        for person in range(N):
            if schedules[person][day] == 'x':
                all_free = False
                break
        if all_free:
            current_streak += 1
            max_days = max(max_days, current_streak)
        else:
            current_streak = 0
    
    return max_days

# Reading input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
D = int(data[1])
schedules = data[2:]

# Calculate and print the result
print(max_consecutive_free_days(N, D, schedules))