import sys
import threading

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    H = int(data[0]); W = int(data[1]); X = int(data[2])
    P = int(data[3]) - 1; Q = int(data[4]) - 1
    # Read slime strengths in row-major order
    Sflat = list(map(int, data[5:]))

    N = H * W
    idx0 = P * W + Q
    T = Sflat[idx0]

    # state arrays: 0 = unvisited, 1 = in frontier or visited
    visited = bytearray(N)
    inFront = bytearray(N)
    visited[idx0] = 1

    import heapq
    heappush = heapq.heappush
    heappop = heapq.heappop

    hq = []
    # Push the initial neighbors of the starting cell
    # Up
    if idx0 >= W:
        ni = idx0 - W
        heappush(hq, (Sflat[ni], ni))
        inFront[ni] = 1
    # Down
    if idx0 < N - W:
        ni = idx0 + W
        heappush(hq, (Sflat[ni], ni))
        inFront[ni] = 1
    # Left
    j0 = idx0 % W
    if j0 > 0:
        ni = idx0 - 1
        heappush(hq, (Sflat[ni], ni))
        inFront[ni] = 1
    # Right
    if j0 < W - 1:
        ni = idx0 + 1
        heappush(hq, (Sflat[ni], ni))
        inFront[ni] = 1

    N_minus_W = N - W
    # Expand by always absorbing the smallest-strength frontier slime
    while hq:
        s, idx = hq[0]
        # Can absorb only if s < T/X  <=>  s*X < T
        if s * X >= T:
            break
        heappop(hq)
        T += s
        visited[idx] = 1

        j = idx % W
        # Up
        if idx >= W:
            ni = idx - W
            if visited[ni] == 0 and inFront[ni] == 0:
                inFront[ni] = 1
                heappush(hq, (Sflat[ni], ni))
        # Down
        if idx < N_minus_W:
            ni = idx + W
            if visited[ni] == 0 and inFront[ni] == 0:
                inFront[ni] = 1
                heappush(hq, (Sflat[ni], ni))
        # Left
        if j > 0:
            ni = idx - 1
            if visited[ni] == 0 and inFront[ni] == 0:
                inFront[ni] = 1
                heappush(hq, (Sflat[ni], ni))
        # Right
        if j < W - 1:
            ni = idx + 1
            if visited[ni] == 0 and inFront[ni] == 0:
                inFront[ni] = 1
                heappush(hq, (Sflat[ni], ni))

    # Output the maximum possible strength
    sys.stdout.write(str(T))

if __name__ == "__main__":
    main()