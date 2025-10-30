import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    W = list(map(int, sys.stdin.readline().split()))
    
    boxes = defaultdict(list)
    for i in range(n):
        box = A[i]
        weight = W[i]
        boxes[box].append(weight)
    
    sum_all = sum(W)
    S = 0
    for j in boxes:
        if boxes[j]:
            S += max(boxes[j])
    
    cost = sum_all - S
    print(cost)

if __name__ == "__main__":
    main()