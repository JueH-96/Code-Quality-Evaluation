import sys
import random

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    A = [0] + [int(x) for x in input[ptr:ptr+N]]
    ptr += N
    B = [0] + [int(x) for x in input[ptr:ptr+N]]
    ptr += N
    queries = []
    for _ in range(Q):
        l = int(input[ptr])
        ptr += 1
        r = int(input[ptr])
        ptr += 1
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1
        queries.append((l, r, L, R))
    
    random.seed(42)
    hash1 = [0] + [random.getrandbits(64) for _ in range(1, N+1)]
    hash2 = [0] + [random.getrandbits(64) for _ in range(1, N+1)]
    
    prefixA1 = [0] * (N + 1)
    prefixA2 = [0] * (N + 1)
    for i in range(1, N+1):
        prefixA1[i] = prefixA1[i-1] + hash1[A[i]]
        prefixA2[i] = prefixA2[i-1] + hash2[A[i]]
    
    prefixB1 = [0] * (N + 1)
    prefixB2 = [0] * (N + 1)
    for i in range(1, N+1):
        prefixB1[i] = prefixB1[i-1] + hash1[B[i]]
        prefixB2[i] = prefixB2[i-1] + hash2[B[i]]
    
    for query in queries:
        l, r, L, R = query
        lenA = r - l + 1
        lenB = R - L + 1
        if lenA != lenB:
            print("No")
            continue
        sumA1 = prefixA1[r] - prefixA1[l-1]
        sumA2 = prefixA2[r] - prefixA2[l-1]
        sumB1 = prefixB1[R] - prefixB1[L-1]
        sumB2 = prefixB2[R] - prefixB2[L-1]
        if sumA1 == sumB1 and sumA2 == sumB2:
            print("Yes")
        else:
            print("No")

if __name__ == '__main__':
    main()