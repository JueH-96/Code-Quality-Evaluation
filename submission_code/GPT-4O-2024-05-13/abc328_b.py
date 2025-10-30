# YOUR CODE HERE
def is_repdigit(num):
    num_str = str(num)
    return all(d == num_str[0] for d in num_str)

def count_repdigit_days(N, days_in_months):
    count = 0
    for month in range(1, N + 1):
        for day in range(1, days_in_months[month - 1] + 1):
            if is_repdigit(month) and is_repdigit(day):
                count += 1
    return count

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
days_in_months = list(map(int, data[1:]))

print(count_repdigit_days(N, days_in_months))