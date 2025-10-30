import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    idx += N
    W = list(map(int, input[idx:idx+N]))
    
    box_dict = defaultdict(list)
    for i in range(N):
        j = A[i]
        box_dict[j].append(W[i])
    
    sum_S = 0
    for j in box_dict:
        if box_dict[j]:
            sum_S += max(box_dict[j])
    
    total_W = sum(W)
    print(total_W - sum_S)

if __name__ == '__main__':
    main()