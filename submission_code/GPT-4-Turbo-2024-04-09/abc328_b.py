def is_repdigit(x):
    s = str(x)
    return all(c == s[0] for c in s)

def count_repdigit_days(N, days_per_month):
    repdigit_count = 0
    for i in range(1, N + 1):
        if is_repdigit(i):
            for j in range(1, days_per_month[i-1] + 1):
                if is_repdigit(j):
                    repdigit_count += 1
    return repdigit_count

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
days_per_month = list(map(int, data[1:]))

result = count_repdigit_days(N, days_per_month)
print(result)