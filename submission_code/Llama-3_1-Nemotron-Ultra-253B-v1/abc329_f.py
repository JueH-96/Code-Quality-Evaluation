import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N, Q = int(input[ptr]), int(input[ptr+1])
    ptr += 2
    C = list(map(int, input[ptr:ptr+N]))
    ptr += N
    boxes = [defaultdict(int) for _ in range(N)]
    distinct = [0] * N
    empty = [False] * N
    for i in range(N):
        color = C[i]
        boxes[i][color] = 1
        distinct[i] = 1
    for _ in range(Q):
        a = int(input[ptr]) - 1
        b = int(input[ptr+1]) - 1
        ptr += 2
        if empty[a]:
            print(distinct[b])
            continue
        # Move all colors from a to b
        for color, cnt in boxes[a].items():
            if boxes[b].get(color, 0) == 0:
                distinct[b] += 1
            boxes[b][color] += cnt
        boxes[a].clear()
        empty[a] = True
        print(distinct[b])

if __name__ == "__main__":
    main()