import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    x = list(map(int, input[ptr:ptr+Q]))
    ptr += Q

    # Compute s array: size of S after each query
    s = [0] * (Q + 2)  # 1-based
    current_s = 0
    S = set()
    for i in range(1, Q+1):
        xi = x[i-1]
        if xi in S:
            S.remove(xi)
            current_s -= 1
        else:
            S.add(xi)
            current_s += 1
        s[i] = current_s

    # Compute prefix sum array
    ps = [0] * (Q + 2)
    for i in range(1, Q+1):
        ps[i] = ps[i-1] + s[i]

    # Collect toggle times for each j
    toggle_times = defaultdict(list)
    for i in range(Q):
        j = x[i]
        toggle_times[j].append(i + 1)  # 1-based query index

    # Compute A for each j
    A = [0] * (N + 1)
    for j in range(1, N + 1):
        tt = toggle_times.get(j, [])
        tt_sorted = sorted(tt)
        tt_with_ends = [0] + tt_sorted + [Q + 1]
        total = 0
        for m in range(len(tt_with_ends) - 1):
            if m % 2 == 1:  # interval where j is in S
                a = tt_with_ends[m]
                b = tt_with_ends[m + 1] - 1
                if a > Q:
                    continue
                if b > Q:
                    b = Q
                if a > b:
                    continue
                total += ps[b] - ps[a - 1]
        A[j] = total

    # Output the result
    print(' '.join(map(str, A[1:N+1])))

if __name__ == '__main__':
    main()