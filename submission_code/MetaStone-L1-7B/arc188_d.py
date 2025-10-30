import sys
import math

def main():
    MOD = 998244353
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    # Check for invalid A or B
    for a in A:
        if a != -1 and not (1 <= a <= N):
            print(0)
            return
    for b in B:
        if b != -1 and not (1 <= b <= N):
            print(0)
            return
    
    fixed_a = [a for a in A if a != -1]
    fixed_b = [b for b in B if b != -1]
    
    # Check for duplicates in fixed_a and fixed_b
    fixed_set_a = set(fixed_a)
    fixed_set_b = set(fixed_b)
    if len(fixed_a) != len(fixed_set_a) or len(fixed_b) != len(fixed_set_b):
        print(0)
        return
    
    # Check for overlapping fixed indices
    for i in range(N):
        if A[i] != -1 and B[i] != -1 and A[i] != B[i]:
            print(0)
            return
    
    # Compute available numbers for a and b
    used_a = set(fixed_a)
    used_b = set(fixed_b)
    available_a = set(range(1, N+1)) - used_a
    available_b = set(range(1, N+1)) - used_b
    
    # Check if available_a and available_b are the same size
    if len(available_a) != len(available_b):
        print(0)
        return
    
    # Compute factorial modulo MOD
    def factorial(n):
        if n == 0:
            return 1
        res = 1
        for i in range(2, n+1):
            res = (res * i) % MOD
        return res
    
    # Compute the number of permutations for a and b
    k = len(available_a)
    if k == 0:
        print(1)
        return
    fact = factorial(k)
    total = (fact * fact) % MOD
    print(total)

if __name__ == '__main__':
    main()