def main():
    import sys
    N = int(sys.stdin.readline())
    r = N % 5
    if r <= 2:
        nearest_station = 5 * (N // 5)
    else:
        nearest_station = 5 * (N // 5 + 1)
        if nearest_station > 100:
            nearest_station = 100
    print(nearest_station)

main()