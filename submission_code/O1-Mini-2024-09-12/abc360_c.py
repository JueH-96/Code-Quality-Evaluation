# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    W = list(map(int, data[N+1:2*N+1]))
    
    box_max = [0] * (N + 1)
    total = 0
    for a, w in zip(A, W):
        total += w
        if w > box_max[a]:
            box_max[a] = w
    sum_max = sum(box_max)
    print(total - sum_max)

if __name__ == "__main__":
    main()