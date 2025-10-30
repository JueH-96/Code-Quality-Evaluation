import bisect
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    
    N = int(input[ptr])
    ptr += 1
    
    X = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    P = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    # Compute prefix sum of P
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + P[i]
    
    Q = int(input[ptr])
    ptr += 1
    
    output = []
    for _ in range(Q):
        L = int(input[ptr])
        R = int(input[ptr+1])
        ptr += 2
        
        left = bisect.bisect_left(X, L)
        right = bisect.bisect_right(X, R) - 1
        
        if left <= right:
            res = prefix[right+1] - prefix[left]
        else:
            res = 0
        output.append(str(res))
    
    print('
'.join(output))

if __name__ == "__main__":
    main()