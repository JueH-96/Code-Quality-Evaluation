def main():
    import sys
    N = int(sys.stdin.readline().strip())
    
    # The water stations are at each multiple of 5 from 0 to 100.
    # Find the multiple of 5 just below (or equal to) N:
    lower = (N // 5) * 5
    
    # Find the multiple of 5 just above N:
    upper = lower + 5
    if upper > 100:
        upper = 100

    # Compare distances to decide the nearest station
    if abs(N - lower) <= abs(N - upper):
        print(lower)
    else:
        print(upper)

# Do not remove this call
main()