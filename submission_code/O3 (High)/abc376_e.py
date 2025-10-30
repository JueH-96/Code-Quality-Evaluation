import sys, heapq

def solve_case(N, K, A, B):
    # K_needed = K - 1 elements other than the current max-A one
    K_needed = K - 1

    # Special-case K = 1  →  choose the single i that minimises A[i]*B[i]
    if K_needed == 0:
        return min(a * b for a, b in zip(A, B))

    # sort by A so that going left→right guarantees non-decreasing max A
    pairs = sorted(zip(A, B), key=lambda x: x[0])

    # low  : max-heap (negated) with the currently smallest K_needed B’s
    # high : min-heap with the remaining (bigger) ones
    low, high = [], []
    sum_low = 0                       # sum of the values stored in  low
    best = 10**30                     # something huge

    for idx, (a, b) in enumerate(pairs):
        # We want the answer that uses this element as the maximal-A item.
        # This is possible once we have seen at least K_needed previous items.
        if idx >= K_needed and len(low) == K_needed:
            cand = a * (b + sum_low)
            if cand < best:
                best = cand

        # insert this element’s B into the data structure
        if len(low) < K_needed:
            heapq.heappush(low, -b)
            sum_low += b
        else:
            # the biggest of the current ‘smallest K_needed’ values
            top_low = -low[0]
            if b < top_low:
                # new value belongs to low; move the previous biggest to high
                heapq.heapreplace(low, -b)   # pop & push in one step
                sum_low += b - top_low
                heapq.heappush(high, top_low)
            else:
                heapq.heappush(high, b)

    return best


def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    t = data[0]
    pos = 1
    out_lines = []

    for _ in range(t):
        N, K = data[pos], data[pos + 1]
        pos += 2
        A = data[pos:pos + N]
        pos += N
        B = data[pos:pos + N]
        pos += N

        out_lines.append(str(solve_case(N, K, A, B)))

    sys.stdout.write('
'.join(out_lines))


if __name__ == "__main__":
    main()