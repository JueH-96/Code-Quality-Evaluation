import sys

def get_depth(X):
    return (X).bit_length() - 1

def compute_subtree(X, K, N):
    if K < 0:
        return 0
    if X > N:
        return 0
    max_possible = N // X
    if max_possible == 0:
        return 0
    e_max = (max_possible).bit_length() - 1
    if K > e_max:
        return 0
    else:
        power = 1 << K
        start = X * power
        if start > N:
            return 0
        end = start + (power - 1)
        end = min(end, N)
        return end - start + 1

def solve():
    input = sys.stdin.read().split()
    T = int(input[0])
    idx = 1
    for _ in range(T):
        N = int(input[idx])
        X = int(input[idx+1])
        K = int(input[idx+2])
        idx += 3
        
        if K < 0:
            print(0)
            continue
        
        d = get_depth(X)
        M = min(K, d)
        count = compute_subtree(X, K, N)
        
        for m in range(1, M + 1):
            A = X >> m
            direction = ( (X >> (m-1)) ) % 2
            C = (A << 1) + (1 - direction)
            required_steps = K - m - 1
            
            if required_steps < 0:
                if m == K:
                    count += 1
                continue
            else:
                contrib = compute_subtree(C, required_steps, N)
                count += contrib
        
        print(count)

if __name__ == "__main__":
    solve()