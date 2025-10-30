# YOUR CODE HERE
from collections import Counter

def count_equal_combinations(n, a):
    count = Counter(a)
    result = 0
    for value, freq in count.items():
        if value == 1:
            result += freq * (freq - 1) // 2
        else:
            result += freq * (freq - 1)
    return result

t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    print(count_equal_combinations(n, a))