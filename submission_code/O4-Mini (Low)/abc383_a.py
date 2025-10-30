def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input())
    prev_time = 0
    water = 0
    
    for _ in range(N):
        t, v = map(int, input().split())
        # Leak for the elapsed time
        elapsed = t - prev_time
        water = max(0, water - elapsed)
        # Add new water
        water += v
        prev_time = t
    
    print(water)

if __name__ == "__main__":
    main()