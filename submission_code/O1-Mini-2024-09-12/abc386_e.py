import sys
import itertools

def solve():
    import sys, itertools
    data = sys.stdin.read().split()
    N, K = int(data[0]), int(data[1])
    A = list(map(int, data[2:2+N]))
    
    if K == 1:
        print(max(A))
    elif K == 2:
        max_xor = 0
        for i in range(N):
            ai = A[i]
            for j in range(i+1, N):
                current_xor = ai ^ A[j]
                if current_xor > max_xor:
                    max_xor = current_xor
        print(max_xor)
    else:
        max_xor = 0
        for comb in itertools.combinations(A, K):
            current_xor = 0
            for num in comb:
                current_xor ^= num
            if current_xor > max_xor:
                max_xor = current_xor
        print(max_xor)