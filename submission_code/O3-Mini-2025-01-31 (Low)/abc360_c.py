def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return

    it = iter(data)
    n = int(next(it))
    # Read initial box assignments and weights
    A = [int(next(it)) for _ in range(n)]
    W = [int(next(it)) for _ in range(n)]

    # Create a list for each box to store the weights of items currently in it.
    # Boxes are numbered from 1 to n, so we use an array of length n+1 (ignoring index 0)
    boxes = [[] for _ in range(n + 1)]
    for i in range(n):
        box_no = A[i]
        boxes[box_no].append(W[i])

    # The idea is to leave one item in each non-empty box without moving it.
    # To minimize the removal cost, in any box with 2 or more items, we keep the heaviest
    # item and consider moving the rest.
    # Also, count the number of boxes that are empty (deficit boxes).
    deficit = 0
    # candidate will store the cost (weight) of each surplus item (i.e. items to be moved)
    candidate = []
    for i in range(1, n + 1):
        if len(boxes[i]) == 0:
            deficit += 1
        elif len(boxes[i]) == 1:
            # Only one item, so we do nothing.
            continue
        else:
            # In boxes with two or more items, choose the heaviest one to remain
            # and mark the rest as movable.
            # To do that, find the maximum weight.
            max_w = max(boxes[i])
            used = False  # will keep exactly one max item and move the others
            for weight in boxes[i]:
                if weight == max_w and not used:
                    used = True
                    continue
                candidate.append(weight)

    # We now have a surplus pool containing items that could be moved.
    # Note that the number of required moves equals the number of deficit boxes.
    # To minimize cost, we choose the candidate items with the smallest weights.
    candidate.sort()
    # Since total items = n, the number of surplus items exactly equals deficit.
    # However, if there are more candidate items than deficit (which is possible if
    # we created surplus from boxes more than needed), we pick the smallest deficit ones.
    total_cost = sum(candidate[:deficit])
    sys.stdout.write(str(total_cost))

if __name__ == "__main__":
    main()