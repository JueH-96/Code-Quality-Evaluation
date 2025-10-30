import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    colors = list(map(int, input().split()))
    # mp[i] is the set of distinct colors currently in box i (1-indexed)
    mp = [set() for _ in range(N+1)]
    for i, c in enumerate(colors, start=1):
        mp[i].add(c)

    out = []
    for _ in range(Q):
        a, b = map(int, input().split())
        if a == b:
            # though problem states a != b, just in case
            out.append(str(len(mp[b])))
            continue
        sa = mp[a]
        sb = mp[b]
        # To optimize, always merge the smaller set into the larger one.
        if len(sa) > len(sb):
            # swap the sets in the mp array so that mp[b] is the larger
            mp[a], mp[b] = sb, sa
            sa, sb = sb, sa
        # now sa is mp[a] (smaller), sb is mp[b] (larger)
        # move all distinct colors from a to b
        for col in sa:
            sb.add(col)
        # clear box a
        sa.clear()
        # report distinct count in box b
        out.append(str(len(sb)))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()