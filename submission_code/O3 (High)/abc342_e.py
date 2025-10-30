import sys
import heapq

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input_data = sys.stdin.buffer.read().split()
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))

    # incoming[B] holds all edges (A,l,d,c,last) that end at station B
    incoming = [[] for _ in range(N + 1)]

    for _ in range(M):
        l  = int(next(it))
        d  = int(next(it))
        k  = int(next(it))
        c  = int(next(it))
        A  = int(next(it))
        B  = int(next(it))
        last = l + (k - 1) * d     # latest departure time for this train sequence
        incoming[B].append((A, l, d, c, last))

    INF = 10 ** 30               # bigger than any possible time in input
    latest = [-1] * (N + 1)      # latest[i] : current best departure time from station i
    latest[N] = INF

    # max-heap implemented with negative keys
    pq = [(-INF, N)]
    while pq:
        t_neg, station = heapq.heappop(pq)
        cur_time = -t_neg
        # Skip if this is an outdated record
        if cur_time != latest[station]:
            continue

        # Relax all edges that end at this station
        for (A, l, d, c, last) in incoming[station]:
            if latest[station] == INF:
                # destination is reachable at any time -> we can take the latest train of this sequence
                t_max = last
            else:
                bound = latest[station] - c          # we must arrive no later than this
                if bound < l:                        # too early, no usable train
                    continue
                t_max = bound if bound < last else last

            # Align to the timetable (largest t <= t_max with correct residue)
            n = (t_max - l) // d
            t_depart = l + n * d
            if t_depart > latest[A]:
                latest[A] = t_depart
                heapq.heappush(pq, (-t_depart, A))

    out_lines = []
    for s in range(1, N):
        if latest[s] == -1:
            out_lines.append("Unreachable")
        else:
            out_lines.append(str(latest[s]))
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()