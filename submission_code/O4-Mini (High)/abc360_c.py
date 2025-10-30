import sys

def main():
    input = sys.stdin.readline
    N = int(input())
    A = list(map(int, input().split()))
    W = list(map(int, input().split()))
    
    # Group item weights by their initial box
    boxes = [[] for _ in range(N+1)]
    for a, w in zip(A, W):
        boxes[a].append(w)
    
    # For each non-empty box, leave the heaviest item and move the rest
    total_cost = 0
    for box in boxes:
        if not box:
            continue
        s = 0
        m = 0
        for w in box:
            s += w
            if w > m:
                m = w
        total_cost += (s - m)
    
    print(total_cost)

if __name__ == "__main__":
    main()