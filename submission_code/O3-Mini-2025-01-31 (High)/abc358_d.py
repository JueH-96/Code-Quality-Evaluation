def main():
    import sys
    import heapq

    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    boxes = [int(next(it)) for _ in range(N)]
    reqs = [int(next(it)) for _ in range(M)]

    # Sort boxes in descending order – larger candy counts come first.
    boxes.sort(reverse=True)
    # Sort requirements in descending order – higher requirements first.
    reqs.sort(reverse=True)

    total_cost = 0
    candidate = []  # This min-heap will hold candidate boxes (their cost equals the candy count).
    j = 0  # Pointer for the boxes list.

    # Process each person (with requirement req) from highest to lowest.
    for req in reqs:
        # Add every box that satisfies: box candy count >= current req.
        while j < N and boxes[j] >= req:
            heapq.heappush(candidate, boxes[j])
            j += 1
        # If no eligible box is available, it's impossible.
        if not candidate:
            sys.stdout.write("-1")
            return
        # Pop the box with the minimum cost among those eligible.
        chosen = heapq.heappop(candidate)
        total_cost += chosen

    sys.stdout.write(str(total_cost))


if __name__ == '__main__':
    main()