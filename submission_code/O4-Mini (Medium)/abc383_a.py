def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    cur = 0
    last_time = 0

    for _ in range(N):
        t, v = map(int, input().split())
        # Leak water for the elapsed time
        elapsed = t - last_time
        cur = max(0, cur - elapsed)
        # Add new water
        cur += v
        last_time = t

    # After the last addition, print the remaining water
    print(cur)

if __name__ == "__main__":
    main()