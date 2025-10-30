import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    W = list(map(int, sys.stdin.readline().split()))
    
    boxes = [[] for _ in range(N + 1)]  # 1-based indexing
    
    for a, w in zip(A, W):
        boxes[a].append(w)
    
    sum_max = 0
    for b in range(1, N + 1):
        if boxes[b]:
            sum_max += max(boxes[b])
    
    total = sum(W)
    print(total - sum_max)

if __name__ == "__main__":
    main()