import sys

def get_divisors(n):
    divisors = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return sorted(divisors)

t = int(sys.stdin.readline())
for _ in range(t):
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    divisors = get_divisors(n)
    max_diff = 0
    for k in divisors:
        groups = []
        for i in range(0, n, k):
            s = sum(a[i:i + k])
            groups.append(s)
        current_max = max(groups)
        current_min = min(groups)
        diff = current_max - current_min
        if diff > max_diff:
            max_diff = diff
    print(max_diff)