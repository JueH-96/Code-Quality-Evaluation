def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    wheels = []
    for _ in range(N):
        C_i = int(input[ptr])
        ptr += 1
        P_i = int(input[ptr])
        ptr += 1
        S_i = list(map(int, input[ptr:ptr + P_i]))
        ptr += P_i
        wheels.append({'cost': C_i, 's_list': S_i})

    max_s = max(max(s for s in wheel['s_list']) for wheel in wheels) if wheels else 0

    # Initialize E with a higher size to handle all possible k + s
    size = M + max_s + 1
    E = [0.0] * (size)

    threshold = 1e-13
    max_iterations = 1000000
    for iteration in range(max_iterations):
        E_new = [0.0] * size
        for k in range(M - 1, -1, -1):
            min_cost = float('inf')
            for wheel in wheels:
                C_i = wheel['cost']
                s_list = wheel['s_list']
                P_i = len(s_list)
                sum_e = 0.0
                for s in s_list:
                    next_k = k + s
                    if next_k >= M:
                        continue
                    if next_k >= len(E):
                        continue
                    sum_e += E[next_k]
                avg = sum_e / P_i if P_i != 0 else 0.0
                cost_i = C_i + avg
                if cost_i < min_cost:
                    min_cost = cost_i
            E_new[k] = min_cost
        # Update E with E_new
        max_diff = 0.0
        for k in range(M):
            diff = abs(E_new[k] - E[k])
            if diff > max_diff:
                max_diff = diff
        if max_diff < threshold:
            break
        for k in range(size):
            if k < M:
                E[k] = E_new[k]
        # Optionally, print progress
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}, E[0] = {E[0]:.10f}")

    # The answer is E[0]
    print("{0:.15f}".format(E[0]))

if __name__ == '__main__':
    main()