# YOUR CODE HERE
import sys
from math import gcd

def min_operations(n, k, a):
    if k == 2:
        return sum(1 for x in a if x % 2 != 0)
    elif k == 3:
        return sum(1 for x in a if x % 3 != 0)
    elif k == 4:
        count_1 = sum(1 for x in a if x % 4 == 1)
        count_2 = sum(1 for x in a if x % 4 == 2)
        count_3 = sum(1 for x in a if x % 4 == 3)
        if count_2 > 0:
            return 1
        elif count_1 > 0:
            return 2
        else:
            return 0
    elif k == 5:
        return sum(1 for x in a if x % 5 != 0)

input = sys.stdin.read
data = input().split()

t = int(data[0])
index = 1
results = []

for _ in range(t):
    n = int(data[index])
    k = int(data[index + 1])
    a = list(map(int, data[index + 2:index + 2 + n]))
    index += 2 + n
    results.append(min_operations(n, k, a))

for result in results:
    print(result)