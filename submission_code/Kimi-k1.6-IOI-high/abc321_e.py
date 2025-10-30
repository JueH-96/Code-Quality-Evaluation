import sys

def count_down(X, K, N):
    if K == 0:
        return 1 if X <= N else 0
    if X > N:
        return 0
    q = N // X
    if q == 0:
        return 0
    k_max = q.bit_length() - 1
    if K > k_max:
        return 0
    pow2 = 1 << K
    start = X * pow2
    if start > N:
        return 0
    end = start + pow2 - 1
    if end <= N:
        return pow2
    else:
        return N - start + 1

def count_up(X, K, N):
    up = 0
    current = X // 2  # first ancestor is parent of X (distance 1)
    a = 1
    while current >= 1:
        if a == K:
            up += 1
        required_distance = K - a - 1
        if required_distance >= 0:
            bit = (X >> (a - 1)) & 1
            other_child = current * 2 + (1 - bit)
            if other_child <= N:
                cnt = count_down(other_child, required_distance, N)
            else:
                cnt = 0
            up += cnt
        current = current // 2
        a += 1
    return up

def main():
    input = sys.stdin.read().split()
    T = int(input[0])
    idx = 1
    for _ in range(T):
        N = int(input[idx])
        X = int(input[idx+1])
        K = int(input[idx+2])
        idx += 3
        if K == 0:
            print(1)
        else:
            down = count_down(X, K, N)
            up = count_up(X, K, N)
            print(down + up)

if __name__ == '__main__':
    main()