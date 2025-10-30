import sys
from collections import deque

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + A[i]
    
    B = [0] * (N + 1)
    
    dq = deque()
    L = 0
    R = 0
    dq.append(0)
    
    while R < N:
        while dq and A[R] >= A[dq[-1]]:
            dq.pop()
        dq.append(R)
        
        while L <= R:
            current_max = A[dq[0]]
            current_sum = prefix[R+1] - prefix[L]
            if current_sum > 2 * current_max:
                for k in range(L, R+1):
                    B[k+1] = current_sum
                L += 1
                if dq[0] < L:
                    dq.popleft()
            else:
                break
        R += 1
    
    print(' '.join(map(str, B[1:])))

if __name__ == '__main__':
    main()