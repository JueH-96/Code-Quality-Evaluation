import sys

def solve():
    N, Q = map(int, input().split())
    C = list(map(int, input().split()))

    boxes = [[] for _ in range(N)]
    for i in range(N):
        boxes[i].append(C[i])

    for _ in range(Q):
        a, b = map(int, input().split())
        a -= 1
        b -= 1

        boxes[b].extend(boxes[a])
        boxes[a].clear()

        print(len(set(boxes[b])))

if __name__ == "__main__":
    solve()