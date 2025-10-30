def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    W = []
    X = []
    idx = 1
    for _ in range(N):
        w_i = int(input_data[idx]); x_i = int(input_data[idx+1])
        idx += 2
        W.append(w_i)
        X.append(x_i)

    max_employees = 0
    # Check each possible UTC start time t from 0 to 23
    for t in range(24):
        current_sum = 0
        for i in range(N):
            local_start = (X[i] + t) % 24
            if 9 <= local_start <= 17:
                current_sum += W[i]
        max_employees = max(max_employees, current_sum)

    print(max_employees)

# Do not forget to call main!
main()