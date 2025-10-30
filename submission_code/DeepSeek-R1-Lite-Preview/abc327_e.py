import math

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    
    INF = float('-inf')
    previous = [INF] * (N + 1)
    current = [INF] * (N + 1)
    
    max_R = -float('inf')
    
    for k in range(1, N + 1):
        current_c = 0.9 ** (k - 1)
        max_prev = INF
        for i in range(1, N + 1):
            if k == 1:
                current[i] = P[i - 1] * 1.0
            else:
                if i >= k:
                    max_prev = max(max_prev, previous[i - 1])
                    current[i] = current_c * P[i - 1] + max_prev
                else:
                    current[i] = INF
        max_sum = max(current[k:N+1])
        sum_weights = 10.0 * (1.0 - 0.9 ** k)
        if sum_weights == 0:
            continue
        R = (max_sum / sum_weights) - (1200.0 / math.sqrt(k))
        if R > max_R:
            max_R = R
        previous, current = current, previous
        current = [INF] * (N + 1)
    print(max_R)

if __name__ == '__main__':
    main()