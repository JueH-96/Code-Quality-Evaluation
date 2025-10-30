import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    W = list(map(int, sys.stdin.readline().split()))
    
    box_weights = defaultdict(list)
    
    for i in range(n):
        box = A[i]
        box_weights[box].append(W[i])
    
    sum_max = 0
    for weights in box_weights.values():
        sum_max += max(weights)
    
    total_weight = sum(W)
    print(total_weight - sum_max)

if __name__ == "__main__":
    main()