import sys
import threading

def main():
    import sys

    input = sys.stdin.readline
    N = int(input())
    A = list(map(int, input().split()))
    W = list(map(int, input().split()))

    # Prepare lists of weights per box
    boxes = [[] for _ in range(N+1)]
    for i in range(N):
        boxes[A[i]].append(W[i])

    total_cost = 0
    # For each box with more than one item,
    # we must move out all but one. To minimize cost,
    # we leave the heaviest and move the rest.
    for j in range(1, N+1):
        items = boxes[j]
        if len(items) > 1:
            s = sum(items)
            mx = max(items)
            total_cost += (s - mx)

    print(total_cost)

if __name__ == "__main__":
    main()