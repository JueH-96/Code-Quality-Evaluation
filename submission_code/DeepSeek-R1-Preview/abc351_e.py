import sys

def compute_sum(points):
    m = len(points)
    if m < 2:
        return 0
    # Compute u and v lists
    u = [x + y for x, y in points]
    v = [x - y for x, y in points]
    # Sort and compute sum of absolute differences for u
    u.sort()
    sum_u = 0
    prefix = 0
    for i in range(len(u)):
        sum_u += u[i] * i - prefix
        prefix += u[i]
    # Sort and compute sum of absolute differences for v
    v.sort()
    sum_v = 0
    prefix = 0
    for i in range(len(v)):
        sum_v += v[i] * i - prefix
        prefix += v[i]
    # Total Chebyshev distance sum is (sum_u + sum_v) // 2
    return (sum_u + sum_v) // 2

def main():
    n = int(sys.stdin.readline())
    group0 = []
    group1 = []
    for _ in range(n):
        x, y = map(int, sys.stdin.readline().split())
        s = (x + y) % 2
        if s == 0:
            group0.append((x, y))
        else:
            group1.append((x, y))
    total = compute_sum(group0) + compute_sum(group1)
    print(total)

if __name__ == "__main__":
    main()