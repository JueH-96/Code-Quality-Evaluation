import sys
from collections import deque

# -------------------- core -------------------- #
def main() -> None:
    input = sys.stdin.readline

    N, M = map(int, input().split())
    S = list(input().strip())
    T = list(input().strip())

    n, m = N, M
    # queue of starting indices that can be "unstamped" right now
    q = deque()
    used = [False] * (n - m + 1)          # whether we have already erased this segment

    # helper: can we erase segment that starts at idx?
    def erasable(idx: int) -> bool:
        for j in range(m):
            ch = S[idx + j]
            if ch != '#' and ch != T[j]:
                return False
        return True

    # initial scan – the only ones that match T exactly
    for i in range(n - m + 1):
        if erasable(i):
            q.append(i)
            used[i] = True

    # BFS style erasing
    while q:
        i = q.popleft()
        for j in range(m):
            pos = i + j
            if S[pos] != '#':
                S[pos] = '#'
                # every segment of length m that covers pos may become erasable now
                lo = max(0, pos - m + 1)
                hi = min(pos, n - m)
                for k in range(lo, hi + 1):
                    if not used[k] and erasable(k):
                        used[k] = True
                        q.append(k)

    print('Yes' if all(c == '#' for c in S) else 'No')


# --------------- run ---------------- #
if __name__ == "__main__":
    main()