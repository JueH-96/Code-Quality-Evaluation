import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    W = list(map(int, data[1+n:1+2*n]))
    
    total = sum(W)
    groups = defaultdict(list)
    for i in range(n):
        box = A[i]
        groups[box].append(W[i])
    
    leave_sum = 0
    for key in groups:
        leave_sum += max(groups[key])
    
    result = total - leave_sum
    print(result)

if __name__ == '__main__':
    main()