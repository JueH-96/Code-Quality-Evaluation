def main():
    n = int(input().strip())
    current_time = 0
    current_water = 0
    for _ in range(n):
        t, v = map(int, input().split())
        elapsed = t - current_time
        current_water = max(0, current_water - elapsed)
        current_water += v
        current_time = t
    print(current_water)

if __name__ == "__main__":
    main()