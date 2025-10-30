# YOUR CODE HERE
def is_repdigit(num):
    str_num = str(num)
    return all(digit == str_num[0] for digit in str_num)

def count_repdigit_dates(N, days):
    count = 0
    for month in range(1, N + 1):
        for day in range(1, days[month - 1] + 1):
            if is_repdigit(month) and is_repdigit(day) and str(month)[0] == str(day)[0]:
                count += 1
    return count

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
D = list(map(int, data[1:]))

print(count_repdigit_dates(N, D))