from collections import deque
import sys

def main():
    input = sys.stdin.readline
    N = int(input().strip())
    S = input().strip()
    T = input().strip()
    # Quick check
    if S.count('W') != T.count('W') or S.count('B') != T.count('B'):
        print(-1)
        return
    if S == T:
        print(0)
        return

    # State: (stones_tuple, gap_index)
    # stones_tuple: length N list of 'W'/'B'
    # gap_index: 0..N, gap sits between stones[gap-1] and stones[gap]
    start = (tuple(S), N)
    goal  = (tuple(T), N)

    # max depth per side
    MAXD = 4

    def expand_bfs(root):
        dist = {root: 0}
        q = deque([root])
        while q:
            stones, gap = q.popleft()
            d = dist[(stones, gap)]
            if d == MAXD:
                continue
            # for each adjacent pair at i,i+1
            for i in range(N-1):
                # remove pair at i,i+1
                # build new stones list without those two
                # then insert at gap, and new gap = i
                # careful indices
                # removal
                # stones is tuple, convert to list for slicing
                # but we can slice
                before = stones[:i]
                pair   = stones[i:i+2]
                after  = stones[i+2:]
                rem = before + after  # length N-2
                # insertion index in rem: old gap
                ins = gap
                # but if gap > i, then after removal the rem list has two fewer before gap
                if gap > i:
                    ins -= 2
                # now insert pair at ins
                new_stones = rem[:ins] + pair + rem[ins:]
                new_gap = i
                st = (new_stones, new_gap)
                if st not in dist:
                    dist[st] = d + 1
                    q.append(st)
        return dist

    # BFS from start and from goal
    d1 = expand_bfs(start)
    d2 = expand_bfs(goal)

    # look for intersection
    ans = None
    for st, dd1 in d1.items():
        if st in d2:
            dd2 = d2[st]
            total = dd1 + dd2
            if ans is None or total < ans:
                ans = total

    print(ans if ans is not None else -1)

if __name__ == "__main__":
    main()