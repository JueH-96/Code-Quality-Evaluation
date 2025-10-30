import bisect
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    
    X = list(map(int, input[ptr:ptr + N]))
    ptr += N
    
    P = list(map(int, input[ptr:ptr + N]))
    ptr += N
    
    # Compute prefix sum
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + P[i]
    
    Q = int(input[ptr])
    ptr += 1
    
    for _ in range(Q):
        L = int(input[ptr])
        R = int(input[ptr+1])
        ptr += 2
        
        left_idx = bisect.bisect_left(X, L)
        right_idx = bisect.bisect_right(X, R)
        print(prefix[right_idx] - prefix[left_idx])

if __name__ == "__main__":
    main()