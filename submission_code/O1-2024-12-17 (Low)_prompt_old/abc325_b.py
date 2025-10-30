def solve():
    import sys

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    W = [0]*N
    X = [0]*N
    idx = 1
    for i in range(N):
        W[i] = int(input_data[idx]); idx+=1
        X[i] = int(input_data[idx]); idx+=1

    # We want to hold a 1-hour meeting such that as many employees as possible
    # can attend. A base i can attend if 9 <= (X_i + T) mod 24 <= 17,
    # where T is the meeting start time in UTC.
    # We'll check all T from 0 to 23 and compute the number of employees
    # that can attend.

    max_employees = 0
    for T in range(24):
        current_sum = 0
        for i in range(N):
            local_start = (X[i] + T) % 24
            if 9 <= local_start <= 17:
                current_sum += W[i]
        max_employees = max(max_employees, current_sum)

    print(max_employees)

def main():
    solve()

if __name__ == "__main__":
    main()