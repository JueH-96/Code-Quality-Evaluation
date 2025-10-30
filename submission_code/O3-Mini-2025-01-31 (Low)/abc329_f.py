def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # Read N and Q
    it = iter(input_data)
    n = int(next(it))
    q = int(next(it))

    # Create box dictionaries:
    # Each box[i] is a dictionary mapping a color to its count in that box.
    boxes = [None] * n
    for i in range(n):
        # Each box initially contains one ball with the given color.
        color = int(next(it))
        boxes[i] = {color: 1}

    # Process each query
    # To avoid TLE, we use the small-to-large merging approach.
    results = []
    for _ in range(q):
        # Query: move all balls from box a to box b.
        # Convert to 0-indexed.
        a = int(next(it)) - 1
        b = int(next(it)) - 1

        # If source box a is empty, simply output answer for box b.
        if not boxes[a]:
            results.append(str(len(boxes[b])))
            continue

        # Merge boxes[a] into boxes[b]. We use the small-to-large strategy:
        # If the dictionary for box a is larger than that of box b,
        # merge box b into box a and then swap the pointers,
        # so that the final dictionary is stored at box b.
        if len(boxes[a]) > len(boxes[b]):
            # Merge box b into box a then swap.
            source = boxes[b]
            target = boxes[a]
            for color, cnt in source.items():
                target[color] = target.get(color, 0) + cnt
            boxes[b] = target
            boxes[a] = {}
        else:
            # Merge box a into box b.
            target = boxes[b]
            source = boxes[a]
            for color, cnt in source.items():
                target[color] = target.get(color, 0) + cnt
            boxes[a] = {}

        # The answer is the number of unique colors in box b.
        results.append(str(len(boxes[b])))

    sys.stdout.write("
".join(results))


if __name__ == '__main__':
    main()