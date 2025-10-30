# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    from collections import defaultdict
    cnt = defaultdict(int)
    S = 0
    cnt[S] += 1
    for a in A:
        S = (S + a) % M
        cnt[S] +=1
    ans = 0
    for r in cnt:
        ans += cnt[r] * (cnt[r] -1)
    print(ans)
    
threading.Thread(target=main).start()