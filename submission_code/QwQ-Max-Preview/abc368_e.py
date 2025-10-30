import bisect
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    X1 = int(input[ptr])
    ptr += 1

    trains = []
    for i in range(M):
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        S = int(input[ptr])
        ptr += 1
        T = int(input[ptr])
        ptr += 1
        trains.append((T, A, B, S, i + 1))  # Sort by T, then store A, B, S, original index

    # Sort trains by arrival time T
    trains.sort()
    sorted_trains = []
    for t in trains:
        T_t, A, B, S, idx = t
        sorted_trains.append((A, B, S, T_t, idx))

    X = [0] * (M + 1)  # X[1..M]
    X[1] = X1

    city_data = defaultdict(lambda: {'t': [], 'xt': [], 'prefix_max': []})

    for train in sorted_trains:
        A_j, B_j, S_j, T_j, original_idx = train

        if original_idx == 1:
            # Add to city B_j's data
            xt_j = X1 + T_j
            data = city_data[B_j]
            data['t'].append(T_j)
            data['xt'].append(xt_j)
            if data['prefix_max']:
                new_max = max(data['prefix_max'][-1], xt_j)
            else:
                new_max = xt_j
            data['prefix_max'].append(new_max)
        else:
            # Compute X_j
            c = A_j
            data_c = city_data.get(c, {'t': [], 'xt': [], 'prefix_max': []})
            t_list = data_c['t']
            max_xt = -float('inf')
            if t_list:
                idx = bisect.bisect_right(t_list, S_j) - 1
                if idx >= 0:
                    max_xt = data_c['prefix_max'][idx]
            X_j = max(0, max_xt - S_j) if max_xt != -float('inf') else 0
            X[original_idx] = X_j

            # Add to B_j's data
            data_b = city_data[B_j]
            xt_j_current = X_j + T_j
            data_b['t'].append(T_j)
            data_b['xt'].append(xt_j_current)
            if data_b['prefix_max']:
                new_max = max(data_b['prefix_max'][-1], xt_j_current)
            else:
                new_max = xt_j_current
            data_b['prefix_max'].append(new_max)

    # Collect output
    output = []
    for i in range(2, M + 1):
        output.append(str(X[i]))
    print(' '.join(output))

if __name__ == '__main__':
    main()