import bisect

def main():
    import sys
    n, m, p = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    b = list(map(int, sys.stdin.readline().split()))
    
    sorted_b = sorted(b)
    prefix = [0] * (m + 1)
    for i in range(1, m + 1):
        prefix[i] = prefix[i - 1] + sorted_b[i - 1]
    
    total = 0
    for ai in a:
        t = p - ai
        if t < 0:
            contrib = m * p
        else:
            cnt = bisect.bisect_right(sorted_b, t)
            sum_less = prefix[cnt]
            contrib = ai * cnt + sum_less + (m - cnt) * p
        total += contrib
    print(total)

if __name__ == "__main__":
    main()