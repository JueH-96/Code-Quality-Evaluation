# YOUR CODE HERE
def solve():
    n, q = map(int, input().split())
    c = list(map(int, input().split()))
    boxes = [[] for _ in range(n)]
    for i in range(n):
        boxes[i].append(c[i])

    for _ in range(q):
        a, b = map(int, input().split())
        a -= 1
        b -= 1
        
        boxes[b].extend(boxes[a])
        boxes[a] = []
        
        print(len(set(boxes[b])))

solve()