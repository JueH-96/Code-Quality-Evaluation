def main():
    import sys
    data = list(map(int, sys.stdin.read().split()))
    N = data[0]
    K = data[1:]
    total = sum(K)
    
    # We'll try all possible ways (up to 2^N) to assign each department to either group A or group B.
    # For each assignment, calculate the sum of people in group A (s). Then group B will have (total - s).
    # We'll keep track of the minimum possible value of max(s, total - s).
    
    min_max_lunch = total  # Initialize with total as an upper bound
    for mask in range(1 << N):
        s = 0
        for i in range(N):
            if mask & (1 << i):
                s += K[i]
        min_max_lunch = min(min_max_lunch, max(s, total - s))
    
    print(min_max_lunch)

# Do not forget to call main
if __name__ == "__main__":
    main()