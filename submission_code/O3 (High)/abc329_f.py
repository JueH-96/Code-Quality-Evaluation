import sys


def main() -> None:
    # Increase recursion limit just in case (not really needed here, but harmless)
    sys.setrecursionlimit(1 << 25)

    data = sys.stdin.buffer.read().split()
    it = iter(data)

    n = int(next(it))
    q = int(next(it))

    # one set for each box, initially containing its single ball colour
    boxes = [{int(next(it))} for _ in range(n)]

    out_lines = []

    for _ in range(q):
        a = int(next(it)) - 1     # 0-based index of source box
        b = int(next(it)) - 1     # 0-based index of destination box

        # shortcut if source is already empty – nothing changes
        if not boxes[a]:
            out_lines.append(str(len(boxes[b])))
            continue

        set_a = boxes[a]
        set_b = boxes[b]

        # always merge the smaller set into the larger one
        # if the larger set is currently in the source box,
        # swap the references so that after the swap `set_a`
        # is the smaller one and `set_b` the larger one,
        # and `boxes[b]` points to the large set.
        if len(set_a) > len(set_b):
            boxes[a], boxes[b] = set_b, set_a
            set_a, set_b = set_b, set_a

        # merge smaller into larger
        set_b.update(set_a)
        set_a.clear()           # source box becomes empty

        out_lines.append(str(len(set_b)))

    sys.stdout.write("
".join(out_lines))


if __name__ == "__main__":
    main()