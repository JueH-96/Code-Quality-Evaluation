import sys
import threading

def main():
    import sys
    from collections import deque

    data = sys.stdin.read().split()
    if not data:
        print(-1)
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    X = [0]*M
    A = [0]*M
    for i in range(M):
        X[i] = int(next(it))
    for i in range(M):
        A[i] = int(next(it))

    # total stones must equal N
    if sum(A) != N:
        print(-1)
        return

    # sort by position
    pairs = sorted(zip(X, A), key=lambda x: x[0])
    queue = deque()  # stores (pos, count) of surplus stones
    cost = 0
    prev_x = 1

    # helper to process a block of 'demands' demands starting at position 'start_pos'
    def fulfill(demands, start_pos):
        nonlocal cost, queue
        cur_pos = start_pos
        while demands > 0:
            if not queue:
                # no supply to fulfill
                print(-1)
                sys.exit(0)
            pos, cnt = queue[0]
            d = cnt if cnt < demands else demands
            # cost: sum_{k=0..d-1} ((cur_pos + k) - pos)
            # = d*(cur_pos - pos) + d*(d-1)/2
            dist0 = cur_pos - pos
            cost += d * dist0 + (d*(d-1))//2
            # update
            demands -= d
            cur_pos += d
            if d == cnt:
                queue.popleft()
            else:
                queue[0] = (pos, cnt - d)

    # process each source cell
    for xi, ai in pairs:
        # first fill empty cells in [prev_x, xi-1]
        if xi > prev_x:
            gap = xi - prev_x
            fulfill(gap, prev_x)
        # now at xi, it has ai stones, needs 1: surplus = ai-1
        surplus = ai - 1
        if surplus > 0:
            queue.append((xi, surplus))
        # next position to consider
        prev_x = xi + 1

    # process trailing empty cells after last source
    if prev_x <= N:
        gap = N - prev_x + 1
        fulfill(gap, prev_x)

    # if we reach here, all demands met
    print(cost)

if __name__ == "__main__":
    main()