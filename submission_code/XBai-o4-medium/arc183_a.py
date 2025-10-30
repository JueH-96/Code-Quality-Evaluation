import sys

def main():
    N, K = map(int, sys.stdin.readline().split())
    
    # Compute K!
    fact_k = 1
    for i in range(1, K+1):
        fact_k *= i
    
    # Compute initial_S which is the total number of good sequences
    initial_S = 1
    for i in range(N):
        m = (N - i) * K
        product = 1
        for j in range(K):
            product *= (m - j)
        term = product // fact_k
        initial_S *= term
    
    target = (initial_S + 1) // 2
    
    remaining = [K] * N
    current_S = initial_S
    sequence = []
    
    for _ in range(N * K):
        total_remaining = N * K - len(sequence)
        for x in range(1, N+1):
            idx = x - 1
            if remaining[idx] == 0:
                continue
            possible = (current_S * remaining[idx]) // total_remaining
            if target <= possible:
                sequence.append(x)
                remaining[idx] -= 1
                current_S = possible
                break
            else:
                target -= possible
    
    print(' '.join(map(str, sequence)))

if __name__ == "__main__":
    main()