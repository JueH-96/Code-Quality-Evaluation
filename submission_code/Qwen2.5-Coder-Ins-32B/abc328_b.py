# YOUR CODE HERE
import sys

def count_repdigit_dates(N, days_in_months):
    def is_repdigit(num):
        s = str(num)
        return all(c == s[0] for c in s)

    count = 0
    for month in range(1, N + 1):
        if not is_repdigit(month):
            continue
        for day in range(1, days_in_months[month - 1] + 1):
            if is_repdigit(day):
                count += 1
    return count

input = sys.stdin.read().split()
N = int(input[0])
days_in_months = list(map(int, input[1:]))

print(count_repdigit_dates(N, days_in_months))