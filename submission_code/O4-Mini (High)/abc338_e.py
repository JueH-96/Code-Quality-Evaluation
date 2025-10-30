import sys
import threading

def main():
    import sys
    data = sys.stdin
    line = data.readline().strip()
    if not line:
        return
    n = int(line)
    # There are 2n points on the circle
    # events[pos] = chord_id whose endpoint is at position pos
    events = [0] * (2*n + 1)
    # visited[ch] = whether we've seen one endpoint of chord ch already
    visited = [False] * (n + 1)
    for i in range(1, n+1):
        l = data.readline().split()
        a = int(l[0]); b = int(l[1])
        # ensure pos1 < pos2 in our linear scan
        if a < b:
            p1, p2 = a, b
        else:
            p1, p2 = b, a
        events[p1] = i
        events[p2] = i

    stack = []
    # scan positions 1..2n
    for pos in range(1, 2*n + 1):
        ch = events[pos]
        # each pos must be an endpoint
        # if first time we see this chord -> opening
        if not visited[ch]:
            visited[ch] = True
            stack.append(ch)
        else:
            # second time -> closing
            if not stack or stack[-1] != ch:
                # top mismatch: crossing detected
                print("Yes")
                return
            # pop the matching open
            stack.pop()

    # no crossings found
    print("No")

if __name__ == "__main__":
    main()