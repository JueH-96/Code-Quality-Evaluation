def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    W = []
    X = []
    idx = 1
    for _ in range(N):
        w, x = map(int, input_data[idx:idx+2])
        idx += 2
        W.append(w)
        X.append(x)
    
    max_employees = 0
    # We only need to check meeting start times T = 0..23 (UTC) because of the 24-hour cycle
    for T in range(24):
        current_sum = 0
        for i in range(N):
            # Local time at base i when it's T in UTC
            local_time_start = (X[i] + T) % 24
            # If local_time_start is between 9 and 17 (inclusive),
            # it means the base's 9:00-18:00 window can accommodate a 1-hour meeting
            if 9 <= local_time_start <= 17:
                current_sum += W[i]
        max_employees = max(max_employees, current_sum)
    
    print(max_employees)

# Don't forget to call main()!
if __name__ == "__main__":
    main()