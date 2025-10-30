import sys
from itertools import permutations

def main():
    N, K = map(int, sys.stdin.readline().split())
    funcs = []
    for _ in range(N):
        A, B = map(int, sys.stdin.readline().split())
        funcs.append((A, B))
    
    # Sort by A + B in descending order, then select top K
    funcs.sort(key=lambda x: -(x[0] + x[1]))
    selected = funcs[:K]
    
    max_total = 0
    for perm in permutations(selected):
        current = 1
        total = 1
        sum_term = 0
        for A, B in perm:
            sum_term = A * sum_term + B * current
            current *= A
        total = current + sum_term
        if total > max_total:
            max_total = total
    print(max_total)

if __name__ == '__main__':
    main()