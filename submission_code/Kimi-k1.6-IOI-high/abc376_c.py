def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N-1]))
    ptr += N-1
    
    A.sort()
    B.sort()
    
    # Compute prefix_ok
    prefix_ok = [False] * N
    prefix_ok[0] = True
    for k in range(1, N):
        if k-1 < len(B):
            condition = (A[k-1] <= B[k-1])
        else:
            condition = False
        prefix_ok[k] = prefix_ok[k-1] and condition
    
    # Compute suffix_ok
    suffix_ok = [False] * N
    suffix_ok[N-1] = True
    for k in range(N-2, -1, -1):
        if k < len(B):
            condition = (A[k+1] <= B[k])
        else:
            condition = False
        suffix_ok[k] = suffix_ok[k+1] and condition
    
    min_x = float('inf')
    for k in range(N):
        if prefix_ok[k] and suffix_ok[k]:
            if k == 0:
                x_candidate = max(A[0], 1)
                if x_candidate <= B[0]:
                    if x_candidate < min_x:
                        min_x = x_candidate
            elif k < N-1:
                lower = B[k-1]
                upper = B[k]
                x_candidate = max(A[k], lower)
                if x_candidate <= upper:
                    if x_candidate < min_x:
                        min_x = x_candidate
            else:
                lower = B[N-2]
                x_candidate = max(A[N-1], lower)
                if x_candidate < min_x:
                    min_x = x_candidate
    
    if min_x == float('inf'):
        print(-1)
    else:
        print(min_x)

if __name__ == '__main__':
    main()