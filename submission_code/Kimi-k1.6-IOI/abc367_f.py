import random

def main():
    import sys
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
    
    max_val = N
    rand1 = [0] * (max_val + 2)  # x can be 1..N
    rand2 = [0] * (max_val + 2)
    random.seed(42)
    for x in range(1, max_val +1):
        rand1[x] = random.getrandbits(64)
        rand2[x] = random.getrandbits(64)
    
    # Compute prefix sums for A
    prefix1_A = [0] * (N +1)
    prefix2_A = [0] * (N +1)
    for i in range(N):
        prefix1_A[i+1] = prefix1_A[i] + rand1[A[i]]
        prefix2_A[i+1] = prefix2_A[i] + rand2[A[i]]
    
    # Compute prefix sums for B
    prefix1_B = [0] * (N +1)
    prefix2_B = [0] * (N +1)
    for i in range(N):
        prefix1_B[i+1] = prefix1_B[i] + rand1[B[i]]
        prefix2_B[i+1] = prefix2_B[i] + rand2[B[i]]
    
    # Process queries
    output = []
    for _ in range(Q):
        l = int(data[ptr])
        ptr +=1
        r = int(data[ptr])
        ptr +=1
        L = int(data[ptr])
        ptr +=1
        R = int(data[ptr])
        ptr +=1
        
        sum_a1 = prefix1_A[r] - prefix1_A[l-1]
        sum_a2 = prefix2_A[r] - prefix2_A[l-1]
        sum_b1 = prefix1_B[R] - prefix1_B[L-1]
        sum_b2 = prefix2_B[R] - prefix2_B[L-1]
        
        if sum_a1 == sum_b1 and sum_a2 == sum_b2:
            output.append("Yes")
        else:
            output.append("No")
    
    print('
'.join(output))

if __name__ == "__main__":
    main()