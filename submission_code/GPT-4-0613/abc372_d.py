# YOUR CODE HERE
import sys
from collections import deque
from itertools import accumulate

def main():
    N = int(sys.stdin.readline())
    H = list(map(int, sys.stdin.readline().split()))
    H = [(h, i) for i, h in enumerate(H)]
    H.sort(reverse=True)

    ans = [0]*N
    acc = [0]*(N+1)
    acc = list(accumulate(acc))
    q = deque()
    for h, i in H:
        while q and q[-1] < i:
            j = q.pop()
            ans[j] = acc[j] - acc[q[-1]+1]
        q.append(i)
        acc[i+1] = acc[i] + 1
    while q:
        j = q.pop()
        ans[j] = acc[j] - acc[q[-1]+1] if q else acc[j]
    print(*ans)

if __name__ == "__main__":
    main()