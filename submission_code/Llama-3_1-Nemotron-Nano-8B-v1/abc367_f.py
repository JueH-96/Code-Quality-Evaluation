import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N, Q = int(input[ptr]), int(input[ptr+1])
    ptr +=2
    
    A = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    B = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    
    # Precompute prefix sums for A
    prefix_sum_A = [0]*(N+1)
    prefix_sq_sum_A = [0]*(N+1)
    prefix_cu_sum_A = [0]*(N+1)
    for i in range(1, N+1):
        a = A[i-1]
        prefix_sum_A[i] = prefix_sum_A[i-1] + a
        prefix_sq_sum_A[i] = prefix_sq_sum_A[i-1] + a*a
        prefix_cu_sum_A[i] = prefix_cu_sum_A[i-1] + a*a*a
    
    # Precompute prefix sums for B
    prefix_sum_B = [0]*(N+1)
    prefix_sq_sum_B = [0]*(N+1)
    prefix_cu_sum_B = [0]*(N+1)
    for i in range(1, N+1):
        b = B[i-1]
        prefix_sum_B[i] = prefix_sum_B[i-1] + b
        prefix_sq_sum_B[i] = prefix_sq_sum_B[i-1] + b*b
        prefix_cu_sum_B[i] = prefix_cu_sum_B[i-1] + b*b*b
    
    for _ in range(Q):
        l = int(input[ptr])
        r = int(input[ptr+1])
        L = int(input[ptr+2])
        R = int(input[ptr+3])
        ptr +=4
        
        len_A = r - l + 1
        len_B = R - L + 1
        if len_A != len_B:
            print("No")
            continue
        
        sum_A = prefix_sum_A[r] - prefix_sum_A[l-1]
        sum_B = prefix_sum_B[R] - prefix_sum_B[L-1]
        if sum_A != sum_B:
            print("No")
            continue
        
        sum_sq_A = prefix_sq_sum_A[r] - prefix_sq_sum_A[l-1]
        sum_sq_B = prefix_sq_sum_B[R] - prefix_sq_sum_B[L-1]
        if sum_sq_A != sum_sq_B:
            print("No")
            continue
        
        sum_cu_A = prefix_cu_sum_A[r] - prefix_cu_sum_A[l-1]
        sum_cu_B = prefix_cu_sum_B[R] - prefix_cu_sum_B[L-1]
        if sum_cu_A == sum_cu_B:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()