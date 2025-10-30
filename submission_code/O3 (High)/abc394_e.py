import sys
from collections import deque

def iter_bits(x):
    """yield indices of set bits in x (least-significant first)"""
    while x:
        lb = x & -x          # lowest set bit
        idx = lb.bit_length() - 1
        yield idx
        x ^= lb

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin.read().splitlines()
    N = int(data[0])
    rows = data[1:]

    ALPHA = 26
    # succ_bits[c][u] : bitset of vertices q s.t. edge u -> q labelled c
    # pred_bits[c][v] : bitset of vertices p s.t. edge p -> v labelled c
    succ_bits = [[0] * N for _ in range(ALPHA)]
    pred_bits = [[0] * N for _ in range(ALPHA)]

    for i in range(N):
        line = rows[i].rstrip()
        for j, ch in enumerate(line):
            if ch != '-':
                c = ord(ch) - 97
                succ_bits[c][i] |= 1 << j
                pred_bits[c][j] |= 1 << i

    INF = 10 ** 9
    dist = [[INF] * N for _ in range(N)]
    visited_row = [0] * N                 # bitset of visited (u, *)
    que = deque()

    # length 0 (empty string)
    for v in range(N):
        dist[v][v] = 0
        visited_row[v] |= 1 << v
        que.append((v, v))

    # length 1 (single edge)
    for c in range(ALPHA):
        for u in range(N):
            mask = succ_bits[c][u]
            for v in iter_bits(mask):
                if (visited_row[u] >> v) & 1:
                    continue
                dist[u][v] = 1
                visited_row[u] |= 1 << v
                que.append((u, v))

    # BFS, every transition adds 2 to the length
    while que:
        u, v = que.popleft()
        nd = dist[u][v] + 2
        for c in range(ALPHA):
            preds = pred_bits[c][u]
            succs = succ_bits[c][v]
            if preds == 0 or succs == 0:
                continue
            pb = preds
            while pb:
                p_bit = pb & -pb
                p = p_bit.bit_length() - 1
                pb ^= p_bit

                rem_q = succs & ~visited_row[p]
                if rem_q == 0:
                    continue
                visited_row[p] |= rem_q
                qb = rem_q
                while qb:
                    q_bit = qb & -qb
                    q = q_bit.bit_length() - 1
                    qb ^= q_bit
                    dist[p][q] = nd
                    que.append((p, q))

    out_lines = []
    for i in range(N):
        row = [str(d if d != INF else -1) for d in dist[i]]
        out_lines.append(' '.join(row))
    print('
'.join(out_lines))

if __name__ == "__main__":
    main()