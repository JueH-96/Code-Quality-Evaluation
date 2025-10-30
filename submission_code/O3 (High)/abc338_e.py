import sys

def main() -> None:
    input = sys.stdin.readline

    N = int(input())
    total_points = 2 * N

    # mapping[p] = id of the chord that has endpoint at point p (1-based)
    mapping = [None] * (total_points + 1)

    for cid in range(N):
        a, b = map(int, input().split())
        mapping[a] = cid
        mapping[b] = cid

    in_stack = [False] * N          # whether the chord is currently "open"
    stack = []                      # stack storing open chord ids

    for p in range(1, total_points + 1):
        cid = mapping[p]

        if not in_stack[cid]:       # first endpoint → open the chord
            in_stack[cid] = True
            stack.append(cid)
        else:                       # second endpoint → close the chord
            # If the chord to be closed is not on top, there is an intersection
            if stack[-1] != cid:
                print("Yes")
                return
            stack.pop()
            in_stack[cid] = False

    # Scanned everything without problems ⇒ no intersections
    print("No")

if __name__ == "__main__":
    main()