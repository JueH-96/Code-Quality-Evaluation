from collections import deque

def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    S = input().strip()
    T = input().strip()

    # Quick check: the number of W and B stones must match
    if S.count('W') != T.count('W') or S.count('B') != T.count('B'):
        print(-1)
        return

    # We'll work on a board of length N+2, with two empty cells represented by '.'
    start = S + ".."
    goal  = T + ".."
    if start == goal:
        print(0)
        return

    # BFS
    dist = {start: 0}
    dq = deque([start])

    while dq:
        cur = dq.popleft()
        d = dist[cur]
        # find empty block
        e = cur.find('.')
        # e should be in [0..N], and cur[e+1]=='.'
        # try every adjacent stone pair x, x+1
        for x in range(0, N+1):
            # ensure x,x+1 in range and both are stones
            if cur[x] != '.' and cur[x+1] != '.':
                # move stones at x,x+1 to e,e+1
                # build new state
                lst = list(cur)
                lst[e], lst[e+1] = lst[x], lst[x+1]
                lst[x], lst[x+1] = '.', '.'
                nxt = ''.join(lst)
                if nxt not in dist:
                    if nxt == goal:
                        print(d+1)
                        return
                    dist[nxt] = d+1
                    dq.append(nxt)
    # if BFS ends without finding goal
    print(-1)

if __name__ == "__main__":
    main()