import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    L = int(input[ptr])
    ptr += 1

    a = list(map(int, input[ptr:ptr+N]))
    ptr += N

    b = list(map(int, input[ptr:ptr+M]))
    ptr += M

    forbidden = set()
    for _ in range(L):
        c = int(input[ptr])
        ptr += 1
        d = int(input[ptr])
        ptr += 1
        forbidden.add((c, d))

    # Prepare sorted lists with original indices
    sorted_a = sorted([(i + 1, val) for i, val in enumerate(a)], key=lambda x: -x[1])
    sorted_b = sorted([(i + 1, val) for i, val in enumerate(b)], key=lambda x: -x[1])

    heap = []
    visited = set()
    if not sorted_a or not sorted_b:
        print(0)
        return

    # Push the first element's negative sum to simulate max-heap
    first_sum_neg = -(sorted_a[0][1] + sorted_b[0][1])
    heapq.heappush(heap, (first_sum_neg, 0, 0))
    visited.add((0, 0))

    while heap:
        sum_neg, i, j = heapq.heappop(heap)
        current_sum = -sum_neg

        a_dish, b_dish = sorted_a[i][0], sorted_b[j][0]
        if (a_dish, b_dish) not in forbidden:
            print(current_sum)
            return

        # Push next possible elements
        ni, nj = i + 1, j
        if ni < len(sorted_a) and (ni, nj) not in visited:
            heapq.heappush(heap, (-(sorted_a[ni][1] + sorted_b[nj][1]), ni, nj))
            visited.add((ni, nj))

        ni, nj = i, j + 1
        if nj < len(sorted_b) and (ni, nj) not in visited:
            heapq.heappush(heap, (-(sorted_a[ni][1] + sorted_b[nj][1]), ni, nj))
            visited.add((ni, nj))

if __name__ == "__main__":
    main()