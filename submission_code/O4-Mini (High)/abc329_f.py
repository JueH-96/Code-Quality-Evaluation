import sys
def main():
    input = sys.stdin.readline
    N, Q = map(int, input().split())
    C = list(map(int, input().split()))
    # boxes[i] will hold the set of colors currently in box i
    boxes = [set() for _ in range(N+1)]
    for i, c in enumerate(C, 1):
        boxes[i].add(c)
    ans = []
    for _ in range(Q):
        a, b = map(int, input().split())
        # Always merge the smaller set into the larger one
        if len(boxes[a]) > len(boxes[b]):
            boxes[a], boxes[b] = boxes[b], boxes[a]
        boxes[b].update(boxes[a])
        boxes[a].clear()
        ans.append(str(len(boxes[b])))
    # Output all answers at once
    print('
'.join(ans))

main()