def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    water = 0
    last_time = 0

    for _ in range(N):
        t, v = map(int, input().split())
        # Leak reduces water by (t - last_time), but not below 0
        water = max(0, water - (t - last_time))
        # Add new water
        water += v
        last_time = t

    print(water)

if __name__ == "__main__":
    main()