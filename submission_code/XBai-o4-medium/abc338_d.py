import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    
    original_sum = 0
    diff = [0] * (N + 2)  # indexes 0 to N+1

    for i in range(M - 1):
        A = X[i]
        B = X[i + 1]
        cw = (B - A) % N
        ccw = N - cw
        if cw < ccw:
            d = cw
            S = A
            E = B
            if S < E:
                L = S
                R = E - 1
                val = (N - 2 * d)
                diff[L] += val
                if R + 1 <= N:
                    diff[R + 1] -= val
            else:
                L1 = S
                R1 = N
                val = (N - 2 * d)
                diff[L1] += val
                diff[R1 + 1] -= val
                if E - 1 >= 1:
                    L2 = 1
                    R2 = E - 1
                    diff[L2] += val
                    if R2 + 1 <= N:
                        diff[R2 + 1] -= val
        elif ccw < cw:
            d = ccw
            S = B
            E = A
            if S < E:
                L = S
                R = E - 1
                val = (N - 2 * d)
                diff[L] += val
                if R + 1 <= N:
                    diff[R + 1] -= val
            else:
                L1 = S
                R1 = N
                val = (N - 2 * d)
                diff[L1] += val
                diff[R1 + 1] -= val
                if E - 1 >= 1:
                    L2 = 1
                    R2 = E - 1
                    diff[L2] += val
                    if R2 + 1 <= N:
                        diff[R2 + 1] -= val
        original_sum += min(cw, ccw)
    
    current = 0
    min_added = float('inf')
    for bridge in range(1, N + 1):
        current += diff[bridge]
        if current < min_added:
            min_added = current
    
    print(original_sum + min_added)

if __name__ == '__main__':
    main()