import sys
import heapq
from typing import List

def decode_prufer(index: int, base: int, length: int) -> List[int]:
    """return the Prufer sequence that corresponds to `index`
       when sequences are ordered lexicographically
       (index written in base `base` with exactly `length` digits)."""
    seq = [0] * length
    for i in range(length - 1, -1, -1):
        seq[i] = index % base
        index //= base
    return seq


def main() -> None:
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))
    K = int(next(it))

    # weight matrix, -1 means "edge absent"
    w = [[-1] * N for _ in range(N)]
    for _ in range(M):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        wt = int(next(it))
        w[u][v] = w[v][u] = wt % K  # only the residue is ever used

    if N == 1:          # degenerate but not allowed by constraints
        print(0)
        return

    # special case N == 2: there is exactly one possible tree
    if N == 2:
        if w[0][1] == -1:
            # graph is guaranteed connected, this should not happen
            print(0)
        else:
            print(w[0][1] % K)
        return

    seq_len = N - 2
    total_sequences = pow(N, seq_len)

    best = K  # larger than any possible modulo value

    for idx in range(total_sequences):
        prufer = decode_prufer(idx, N, seq_len)

        cnt = [1] * N
        for v in prufer:
            cnt[v] += 1

        leaves = [v for v in range(N) if cnt[v] == 1]
        heapq.heapify(leaves)

        cost = 0
        ok = True

        for v in prufer:
            u = heapq.heappop(leaves)

            edge_wt = w[u][v]
            if edge_wt == -1:
                ok = False
                break
            cost += edge_wt
            if cost >= K:
                cost -= K  # equivalent to cost %= K but faster for small overflow

            cnt[v] -= 1
            if cnt[v] == 1:
                heapq.heappush(leaves, v)

        if not ok:
            continue

        # two vertices remain
        u = heapq.heappop(leaves)
        v = heapq.heappop(leaves)

        edge_wt = w[u][v]
        if edge_wt == -1:
            continue  # edge absent -> not a valid tree

        cost += edge_wt
        if cost >= K:
            cost -= K

        if cost < best:
            best = cost
            if best == 0:          # cannot do better
                break

    print(best)


if __name__ == "__main__":
    main()