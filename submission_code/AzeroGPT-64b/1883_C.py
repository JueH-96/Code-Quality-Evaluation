from math import gcd
import sys

def find_min_operations(n, k, arr):
    operations = 0
    for element in arr:
        if gcd(element, k) != k:
            operations += 1
    if k == 2:
        return operations - arr.count(1)
    if k == 3:
        return operations - arr.count(1) - arr.count(2) - arr.count(4)
    if k == 4:
        count_1 = arr.count(1)
        count_2 = arr.count(2)
        count_3 = arr.count(3)
        count_5 = arr.count(5)
        count_7 = arr.count(7)
        count_9 = arr.count(9)
        total = count_1 + count_2 + count_3 + count_5 + count_7 + count_9
        return operations - (total - min(2, count_1) - min(1, count_3) if count_3 >= 1 else total - min(2, count_1))
    if k == 5:
        count_1 = arr.count(1)
        count_2 = arr.count(2)
        count_3 = arr.count(3)
        count_4 = arr.count(4)
        total = count_1 + count_2 + count_3 + count_4
        return operations - (total - min(2, count_1) - min(1, count_2) if count_2 >= 1 else total - min(2, count_1))

t = int(input().strip())
for _ in range(t):
    n, k = map(int, input().strip().split())
    arr = list(map(int, input().strip().split()))
    result = find_min_operations(n, k, arr)
    print(result)