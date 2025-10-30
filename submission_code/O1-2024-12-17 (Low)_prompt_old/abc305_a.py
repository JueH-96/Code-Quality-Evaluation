def solve():
    import sys
    
    N = int(sys.stdin.readline())
    
    # The water stations are located at multiples of 5 from 0 to 100 inclusive.
    # We'll find the multiple of 5 just below (or equal to) N, and the one just above.
    
    lower_station = (N // 5) * 5
    upper_station = lower_station + 5
    
    # Ensure they are within the valid range [0, 100].
    if lower_station < 0:
        lower_station = 0
    if upper_station > 100:
        upper_station = 100
    
    # Compare which station is closer.
    if abs(N - lower_station) <= abs(N - upper_station):
        print(lower_station)
    else:
        print(upper_station)

def main():
    solve()

if __name__ == "__main__":
    main()