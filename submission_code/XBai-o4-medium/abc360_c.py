import sys

def main():
    n = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    W = list(map(int, sys.stdin.readline().split()))
    
    box_max = [0] * (n + 1)  # box 1..n
    
    for i in range(n):
        a = A[i]
        w = W[i]
        if w > box_max[a]:
            box_max[a] = w
    
    total_w = sum(W)
    sum_max = sum(box_max[1:])  # sum boxes 1..n
    
    print(total_w - sum_max)

if __name__ == "__main__":
    main()