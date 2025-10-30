# YOUR CODE HERE
def count_repdigits(N, days_in_month):
    repdigit_count = 0
    
    for month in range(1, N + 1):
        month_str = str(month)
        for day in range(1, days_in_month[month - 1] + 1):
            day_str = str(day)
            if month_str == day_str:
                repdigit_count += 1
                
    return repdigit_count

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
D = list(map(int, data[1:N + 1]))

result = count_repdigits(N, D)
print(result)