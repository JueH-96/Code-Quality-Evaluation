from collections import Counter

def min_operations(n, k, a):
    counter = Counter(i % k for i in a)
    return sum((v + 1) // 2 for v in counter.values())

t = int(input().strip())
for _ in range(t):
    n, k = map(int, input().strip().split())
    a = list(map(int, input().strip().split()))
    print(min_operations(n, k, a))