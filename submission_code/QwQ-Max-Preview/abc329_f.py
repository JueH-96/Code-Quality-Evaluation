import sys
from sys import stdin

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    C = list(map(int, input[ptr:ptr+N]))
    ptr += N

    # Initialize boxes: 1-based index
    boxes = [None] * (N + 1)
    for i in range(1, N+1):
        boxes[i] = {C[i-1]}
    
    for _ in range(Q):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        if len(boxes[a]) <= len(boxes[b]):
            boxes[b].update(boxes[a])
            boxes[a].clear()
            print(len(boxes[b]))
        else:
            boxes[a].update(boxes[b])
            boxes[b].clear()
            boxes[a], boxes[b] = boxes[b], boxes[a]
            print(len(boxes[b]))

if __name__ == "__main__":
    main()