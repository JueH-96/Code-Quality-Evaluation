import sys
import random

def main():
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr +=1
    Q = int(data[ptr])
    ptr +=1
    
    A = list(map(int, data[ptr:ptr+N]))
    ptr +=N
    B = list(map(int, data[ptr:ptr+N]))
    ptr +=N
    
    # Seed for deterministic results
    random.seed(42)
    max_val = N
    r = [0] * (max_val + 2)  # indices 0..max_val
    s = [0] * (max_val + 2)
    for x in range(1, max_val +1):
        r[x] = random.getrandbits(64)
        s[x] = random.getrandbits(64)
    
    # Precompute prefix sums for A and B
    prefix_r_A = [0]*(N+1)
    prefix_s_A = [0]*(N+1)
    for i in range(1, N+1):
        prefix_r_A[i] = prefix_r_A[i-1] + r[A[i-1]]
        prefix_s_A[i] = prefix_s_A[i-1] + s[A[i-1]]
    
    prefix_r_B = [0]*(N+1)
    prefix_s_B = [0]*(N+1)
    for i in range(1, N+1):
        prefix_r_B[i] = prefix_r_B[i-1] + r[B[i-1]]
        prefix_s_B[i] = prefix_s_B[i-1] + s[B[i-1]]
    
    # Process queries
    for _ in range(Q):
        l = int(data[ptr])
        ptr +=1
        r_q = int(data[ptr])
        ptr +=1
        L = int(data[ptr])
        ptr +=1
        R = int(data[ptr])
        ptr +=1
        
        len_a = r_q - l +1
        len_b = R - L +1
        if len_a != len_b:
            print("No")
            continue
        
        sum_r_a = prefix_r_A[r_q] - prefix_r_A[l-1]
        sum_r_b = prefix_r_B[R] - prefix_r_B[L-1]
        sum_s_a = prefix_s_A[r_q] - prefix_s_A[l-1]
        sum_s_b = prefix_s_B[R] - prefix_s_B[L-1]
        
        if sum_r_a == sum_r_b and sum_s_a == sum_s_b:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()