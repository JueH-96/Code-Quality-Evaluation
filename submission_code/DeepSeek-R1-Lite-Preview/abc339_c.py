def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    S = 0
    min_S = float('inf')
    sum_A = 0
    for a in A:
        S += a
        if S < min_S:
            min_S = S
        sum_A = S  # sum_A will be the last S, which is sum of all A_i
    
    P = max(0, -min_S)
    current_passengers = P + sum_A
    print(current_passengers)

if __name__ == "__main__":
    main()