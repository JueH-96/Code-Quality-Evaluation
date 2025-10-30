import sys
import random

random.seed(42)

def main():
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1
    A = list(map(int, data[ptr:ptr+N]))
    ptr += N
    B = list(map(int, data[ptr:ptr+N]))
    ptr += N
    
    # Generate a random 64-bit number for each x in 1..N
    h = {}
    for x in range(1, N+1):
        h[x] = random.getrandbits(64)
    
    # Precompute prefix hashes for A and B (1-based)
    preHashA = [0] * (N + 1)
    for i in range(1, N+1):
        preHashA[i] = preHashA[i-1] + h[A[i-1]]  # A is 0-based
    
    preHashB = [0] * (N + 1)
    for i in range(1, N+1):
        preHashB[i] = preHashB[i-1] + h[B[i-1]]  # B is 0-based
    
    for _ in range(Q):
        l = int(data[ptr])
        ptr += 1
        r = int(data[ptr])
        ptr += 1
        L = int(data[ptr])
        ptr += 1
        R = int(data[ptr])
        ptr += 1
        
        # Convert to 0-based indices for Python lists
        l -= 1
        r -= 1
        L -= 1
        R -= 1
        
        lenA = r - l + 1
        lenB = R - L + 1
        if lenA != lenB:
            print("No")
            continue
        
        # Calculate hash for A's range [l, r] and B's range [L, R]
        hashA = preHashA[r + 1] - preHashA[l]
        hashB = preHashB[R + 1] - preHashB[L]
        
        if hashA == hashB:
            print("Yes")
        else:
            print("No")

if __name__ == '__main__':
    main()