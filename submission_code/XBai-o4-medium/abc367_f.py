import sys
import random

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N, Q = int(input[ptr]), int(input[ptr+1])
    ptr += 2
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    max_val = N
    
    # Generate h1 and h2 for values 1..N
    h1 = [0] * (max_val + 2)  # 1-based up to N
    h2 = [0] * (max_val + 2)
    
    # Seed for h1
    random.seed(42)
    for i in range(1, max_val + 1):
        h1[i] = random.getrandbits(64)
    # Seed for h2
    random.seed(57)
    for i in range(1, max_val + 1):
        h2[i] = random.getrandbits(64)
    
    # Compute prefix hashes for A
    prefixA1 = [0] * (N + 1)
    prefixA2 = [0] * (N + 1)
    for i in range(1, N + 1):
        prefixA1[i] = prefixA1[i-1] + h1[A[i-1]]
        prefixA2[i] = prefixA2[i-1] + h2[A[i-1]]
    
    # Compute prefix hashes for B
    prefixB1 = [0] * (N + 1)
    prefixB2 = [0] * (N + 1)
    for i in range(1, N + 1):
        prefixB1[i] = prefixB1[i-1] + h1[B[i-1]]
        prefixB2[i] = prefixB2[i-1] + h2[B[i-1]]
    
    # Process queries
    output = []
    for _ in range(Q):
        l = int(input[ptr])
        r = int(input[ptr+1])
        L = int(input[ptr+2])
        R = int(input[ptr+3])
        ptr += 4
        
        # Check length
        len_a = r - l + 1
        len_b = R - L + 1
        if len_a != len_b:
            output.append("No")
            continue
        
        a_hash1 = prefixA1[r] - prefixA1[l-1]
        a_hash2 = prefixA2[r] - prefixA2[l-1]
        b_hash1 = prefixB1[R] - prefixB1[L-1]
        b_hash2 = prefixB2[R] - prefixB2[L-1]
        
        if a_hash1 == b_hash1 and a_hash2 == b_hash2:
            output.append("Yes")
        else:
            output.append("No")
    
    print('
'.join(output))

if __name__ == '__main__':
    main()