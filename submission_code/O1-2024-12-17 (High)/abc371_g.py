def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    # Read P (making it 0-based)
    P = [int(x) - 1 for x in data[1:N+1]]
    # Read A (keep values as-is; they are distinct 1..N in some order)
    A = [int(x) for x in data[N+1:2*N+1]]

    visited = [False]*N
    answer = [0]*N  # final arrangement

    for start in range(N):
        if not visited[start]:
            # Collect the current cycle
            cycle = []
            cur = start
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = P[cur]

            L = len(cycle)
            if L == 1:
                # Single-element cycle; it doesn't move at all
                idx = cycle[0]
                answer[idx] = A[idx]
            else:
                # We want the shift k that puts the minimum A-value into
                # the smallest index of the cycle.
                # T[i] = A at cycle[i]
                T = [A[pos] for pos in cycle]
                # x = index of min value in T
                x = min(range(L), key=lambda i: T[i])
                # i0 = where the cycle's smallest position sits
                i0 = min(range(L), key=lambda i: cycle[i])
                # k = (i0 - x) mod L
                k = (i0 - x) % L

                # Fill final arrangement for this cycle
                for i in range(L):
                    answer[cycle[i]] = T[(i - k) % L]

    print(" ".join(map(str, answer)))

# Do not forget to call main()
if __name__ == "__main__":
    main()