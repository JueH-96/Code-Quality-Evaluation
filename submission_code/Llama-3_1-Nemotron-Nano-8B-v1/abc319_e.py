def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N, X, Y = map(int, input[idx:idx+3])
    idx +=3
    steps = []
    for _ in range(N-1):
        P, T = map(int, input[idx:idx+2])
        idx +=2
        steps.append((P, T))
    
    # Precompute for each step and remainder
    # Process steps in forward order
    dp_total_time = [0] * 840
    dp_final_r = list(range(840))
    
    for P, T in steps:
        new_dp_time = [0] * 840
        new_dp_r = [0] * 840
        for r in range(840):
            departure = ((r + P - 1) // P) * P
            time_add = departure + T - r
            new_r = (departure + T) % 840
            new_dp_time[r] = dp_total_time[r] + time_add
            new_dp_r[r] = new_r
        dp_total_time, dp_final_r = new_dp_time, new_dp_r
    
    Q = int(input[idx])
    idx +=1
    for _ in range(Q):
        q = int(input[idx])
        idx +=1
        t0 = q + X
        r = t0 % 840
        res = t0 + dp_total_time[r] + Y
        print(res)

if __name__ == '__main__':
    main()