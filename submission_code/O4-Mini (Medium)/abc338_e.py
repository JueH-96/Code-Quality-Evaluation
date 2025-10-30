import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    total = 2 * n
    # pos[k] = chord index whose endpoint is at position k
    pos = [0] * (total + 1)
    for chord_idx in range(1, n + 1):
        a = int(next(it))
        b = int(next(it))
        pos[a] = chord_idx
        pos[b] = chord_idx

    visited = [False] * (n + 1)
    stack = []
    for k in range(1, total + 1):
        cid = pos[k]
        if not visited[cid]:
            # first time we meet this chord: treat as "opening"
            visited[cid] = True
            stack.append(cid)
        else:
            # second time: should close the most recent open chord
            if not stack or stack[-1] != cid:
                print("Yes")
                return
            stack.pop()

    # If we never saw a crossing
    print("No")

if __name__ == "__main__":
    main()