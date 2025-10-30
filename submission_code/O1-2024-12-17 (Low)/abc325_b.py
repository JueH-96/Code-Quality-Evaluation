def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    W = [int(x) for x in input_data[1::2]]
    X = [int(x) for x in input_data[2::2]]
    
    # We will consider all possible meeting start times T in UTC (0 through 23).
    # For a base i, the local time at T is (X[i] + T) % 24.
    # The base's employees can attend if that local time is in [9..17].
    # We compute the number of employees that can attend for each T and take the maximum.
    
    max_employees = 0
    for T in range(24):
        current_sum = 0
        for i in range(N):
            local_time = (X[i] + T) % 24
            if 9 <= local_time <= 17:
                current_sum += W[i]
        max_employees = max(max_employees, current_sum)
    
    print(max_employees)

if __name__ == "__main__":
    main()