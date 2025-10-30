import sys
import threading
def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    if not line:
        print("No")
        return
    N = int(line[0]); K = int(line[1])
    n = N * K
    # Read edges
    adj = [[] for _ in range(n+1)]
    for _ in range(n-1):
        u,v = map(int, data.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    # Special trivial case
    # We handle K>=1 uniformly below; K=1 will always succeed.
    # Build BFS tree from root 1
    parent = [0] * (n+1)
    children = [[] for _ in range(n+1)]
    from collections import deque
    dq = deque()
    dq.append(1)
    parent[1] = -1  # mark root
    order = []
    while dq:
        u = dq.popleft()
        order.append(u)
        for w in adj[u]:
            if parent[w] == 0:
                parent[w] = u
                children[u].append(w)
                dq.append(w)
    # Prepare DP array
    # dp[u] = 0 means no pending incomplete path at u
    # dp[u] >0 means there is an incomplete path of dp[u] vertices ending at u
    dp = [0] * (n+1)
    possible = True
    # Process nodes in reverse BFS order (children before parent)
    for u in reversed(order):
        # Gather child incompletes
        cnt = 0
        # We can hold at most two child incompletes
        x = y = 0
        for v in children[u]:
            dv = dp[v]
            if dv < 0:
                possible = False
                break
            if dv > 0:
                # an incomplete segment from child v
                cnt += 1
                if cnt == 1:
                    x = dv
                elif cnt == 2:
                    y = dv
                else:
                    # more than two incompletes => impossible
                    possible = False
                    break
        if not possible:
            break
        # Decide dp[u] based on cnt
        if cnt > 2:
            # should not happen due to break above
            possible = False
            break
        if cnt == 2:
            # need to match two segments at u to form one complete path
            if x + y == K - 1:
                # one full path ends at u, no pending upward
                dp[u] = 0
            else:
                possible = False
                break
        elif cnt == 1:
            # one segment to extend
            if x + 1 < K:
                dp[u] = x + 1
            elif x + 1 == K:
                # completes here
                dp[u] = 0
            else:
                # overshoots
                possible = False
                break
        else:
            # cnt == 0: start a new segment at u
            if K == 1:
                # immediately completes
                dp[u] = 0
            else:
                dp[u] = 1
    # After DP
    if not possible or dp[1] != 0:
        print("No")
    else:
        print("Yes")

if __name__ == "__main__":
    # Avoid recursion issues; our code is iterative
    main()