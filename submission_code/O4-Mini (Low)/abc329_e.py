import sys
import threading
from collections import deque

def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    S = input().strip()
    T = input().strip()

    # Number of windows: N - M + 1
    W = N - M + 1

    # need[i]: number of mismatches in window starting at i
    need = [0] * W
    for i in range(W):
        cnt = 0
        for j in range(M):
            if S[i + j] != T[j]:
                cnt += 1
        need[i] = cnt

    done = [False] * W        # whether we've "unstamped" this window
    vis = [False] * N         # whether a position is reverted to '#'
    q = deque()

    # Initialize queue with all windows that perfectly match T (need == 0)
    for i in range(W):
        if need[i] == 0:
            done[i] = True
            q.append(i)
            # Mark their covered positions as visited
            for j in range(M):
                pos = i + j
                if not vis[pos]:
                    vis[pos] = True

    # BFS: for each reverted position, reduce need in overlapping windows
    while q:
        i = q.popleft()
        # for each position in this window
        for j in range(M):
            pos = i + j
            # consider windows that cover pos: start from pos-M+1 to pos
            start = max(0, pos - M + 1)
            end = min(pos, W - 1)
            for w in range(start, end + 1):
                if done[w]:
                    continue
                # If S[w + (pos-w)] was matching T and now we revert it,
                # that reduces the mismatch count by 1 if it was previously mismatched.
                # But easier: whenever any char in window gets reverted, we know
                # that position no longer enforces a mismatch.
                # So we check if S[pos] != T[pos-w], then this mismatch is removed.
                if S[pos] != T[pos - w]:
                    need[w] -= 1
                    if need[w] == 0:
                        done[w] = True
                        q.append(w)
                        # mark all positions of this window
                        for k in range(M):
                            p2 = w + k
                            if not vis[p2]:
                                vis[p2] = True

    # If all positions are visited (i.e., can be stamped over), answer Yes
    if all(vis):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()