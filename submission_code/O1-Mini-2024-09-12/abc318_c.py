# YOUR CODE HERE
def main():
    import sys
    import math

    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    D = int(data[1])
    P = int(data[2])
    F = list(map(int, data[3:3+N]))
    
    sorted_F = sorted(F, reverse=True)
    prefix_sum = [0] * (N +1)
    for i in range(1, N+1):
        prefix_sum[i] = prefix_sum[i-1] + sorted_F[i-1]
    
    sum_all = prefix_sum[N]
    K_max = (N + D -1) // D
    min_cost = sum_all  # K=0 case

    for K in range(1, K_max+1):
        C = K * D
        if C > N:
            C = N
        total_cost = K * P + (sum_all - prefix_sum[C])
        if total_cost < min_cost:
            min_cost = total_cost
    
    print(min_cost)

if __name__ == "__main__":
    main()