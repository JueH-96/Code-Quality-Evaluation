import sys
import threading

def main():
    import sys
    from collections import deque

    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    S = next(it).strip()
    T = next(it).strip()

    # Number of windows
    W = N - M + 1

    # bad_count[i]: number of positions j in window i where S[i+j] != T[j]
    bad_count = [0] * W
    for i in range(W):
        cnt = 0
        # compare window S[i:i+M] to T
        si = S[i:i+M]
        # M is small (<=5)
        for j in range(M):
            if si[j] != T[j]:
                cnt += 1
        bad_count[i] = cnt

    used = [False] * W
    q = deque()
    # enqueue all windows that are already "removable" (no mismatches)
    for i in range(W):
        if bad_count[i] == 0:
            used[i] = True
            q.append(i)

    # current characters, start from S, we'll turn them to '#'
    cur = list(S)
    remaining = N  # count of non-# characters left

    while q:
        i = q.popleft()
        # "remove" window i: turn its M chars to '#'
        for j in range(i, i + M):
            if cur[j] != '#':
                # remove this character
                orig_char = cur[j]
                cur[j] = '#'
                remaining -= 1
                # This removal may fix mismatches in other windows covering j
                # windows k from max(0, j-M+1) to min(j, W-1)
                lo = j - M + 1
                if lo < 0:
                    lo = 0
                hi = j
                if hi > W - 1:
                    hi = W - 1
                # update each window k
                for k in range(lo, hi + 1):
                    if not used[k]:
                        # In window k, position j corresponds to T index (j-k)
                        # If S[j] != T[j-k], that was a bad spot; now it's '#' so it's allowed
                        if S[j] != T[j - k]:
                            bad_count[k] -= 1
                            if bad_count[k] == 0:
                                used[k] = True
                                q.append(k)

    # If we've removed all chars, it's possible
    print("Yes" if remaining == 0 else "No")

if __name__ == "__main__":
    main()