import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    wheels = []
    for _ in range(N):
        parts = list(map(int, sys.stdin.readline().split()))
        C_i = parts[0]
        P_i = parts[1]
        S_list = parts[2:2+P_i]
        wheels.append( (C_i, P_i, S_list) )
    
    E = [0.0] * M  # E[s] for s from 0 to M-1
    
    for s in range(M-1, -1, -1):
        min_val = float('inf')
        for (C_i, P_i, S_list) in wheels:
            count_zero = 0
            sum_non_zero = 0.0
            for S in S_list:
                if S == 0:
                    count_zero += 1
                else:
                    new_s = s + S
                    if new_s < M:
                        sum_non_zero += E[new_s]
            numerator = C_i * P_i + sum_non_zero
            denominator = P_i - count_zero
            candidate = numerator / denominator
            if candidate < min_val:
                min_val = candidate
        E[s] = min_val
    
    print("{0:.12f}".format(E[0]))

if __name__ == "__main__":
    main()