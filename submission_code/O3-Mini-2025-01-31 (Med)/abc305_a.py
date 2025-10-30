def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    
    # Compute the lower water station, which is the multiple of 5 that is not greater than N.
    lower = (N // 5) * 5
    
    # If Takahashi is exactly at a water station, then that station is the nearest.
    if N % 5 == 0:
        print(N)
        return
        
    # Otherwise, the candidate stations are at positions lower and lower + 5.
    # There is a special check when lower == 100 (i.e. at the goal).
    if lower == 100:
        print(100)
        return
        
    # Calculate distances.
    dist_lower = abs(N - lower)
    dist_upper = abs((lower + 5) - N)
    
    # According to the problem, the nearest water station is uniquely determined.
    # So, we simply select the one with the shorter distance.
    if dist_lower <= dist_upper:
        print(lower)
    else:
        print(lower + 5)

if __name__ == '__main__':
    main()