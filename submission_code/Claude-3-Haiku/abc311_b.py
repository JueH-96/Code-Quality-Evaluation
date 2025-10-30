from typing import List

def max_consecutive_free_days(schedules: List[str]) -> int:
    n = len(schedules)
    d = len(schedules[0])
    
    # Check if all people are free on any day
    max_free_days = 0
    for i in range(d):
        all_free = True
        for j in range(n):
            if schedules[j][i] == 'x':
                all_free = False
                break
        if all_free:
            max_free_days = max(max_free_days, 1)
    
    # Check for consecutive free days
    for i in range(d - 1):
        all_free = True
        for j in range(n):
            if schedules[j][i] == 'x' or schedules[j][i + 1] == 'x':
                all_free = False
                break
        if all_free:
            max_free_days = max(max_free_days, 2)
    
    return max_free_days

# Read input
n, d = map(int, input().split())
schedules = [input() for _ in range(n)]

# Solve the problem
print(max_consecutive_free_days(schedules))