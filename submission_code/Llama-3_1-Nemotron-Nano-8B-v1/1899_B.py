import sys
import math

def get_divisors(n):
    divisors = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return sorted(divisors)

t = int(sys.stdin.readline())
for _ in range(t):
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    prefix = [0]
    current = 0
    for num in a:
        current += num
        prefix.append(current)
    divisors = get_divisors(n)
    max_diff = 0
    for k in divisors:
        m = n // k
        current_max = -float('inf')
        current_min = float('inf')
        for i in range(m):
            start = i * k
            end = start + k
            s = prefix[end] - prefix[start]
            if s > current_max:
                current_max = s
            if s < current_min:
                current_min = s
        max_diff = max(max_diff, current_max - current_min)
    print(max_diff)