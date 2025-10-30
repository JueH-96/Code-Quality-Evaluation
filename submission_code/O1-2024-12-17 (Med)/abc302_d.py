def main():
    import sys
    from bisect import bisect_right

    data = sys.stdin.read().strip().split()
    N, M, D = map(int, data[:3])
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:]))

    B.sort()

    answer = -1
    for a in A:
        # Find the largest b such that b <= a + D
        up = a + D
        idx = bisect_right(B, up) - 1
        if idx >= 0:
            b = B[idx]
            # Check if |a - b| <= D --> b >= a - D
            if b >= a - D:
                s = a + b
                if s > answer:
                    answer = s

    print(answer)

# Do not forget to call main()
main()