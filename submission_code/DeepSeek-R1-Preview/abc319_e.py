def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    X = int(input[idx])
    idx += 1
    Y = int(input[idx])
    idx += 1
    
    P_list = []
    T_list = []
    for _ in range(N-1):
        P = int(input[idx])
        idx += 1
        T = int(input[idx])
        idx += 1
        P_list.append(P)
        T_list.append(T)
    
    # Precompute total_time
    total_time = list(range(840))
    for P, T in zip(P_list, T_list):
        total_time = [((t + P - 1) // P) * P + T for t in total_time]
    
    Q = int(input[idx])
    idx += 1
    for _ in range(Q):
        q = int(input[idx])
        idx += 1
        t_start = q + X
        m = t_start % 840
        arrival = t_start + (total_time[m] - m) + Y
        print(arrival)
    
if __name__ == '__main__':
    main()