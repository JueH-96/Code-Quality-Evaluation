import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    P = [int(next(it)) - 1 for _ in range(N)]
    visited = [False] * N
    ans = [0] * N
    # Memoize pow(2, K, l) for seen cycle lengths
    mod_cache = {}
    for i in range(N):
        if not visited[i]:
            # Build the cycle starting from i
            cycle = []
            x = i
            while not visited[x]:
                visited[x] = True
                cycle.append(x)
                x = P[x]
            l = len(cycle)
            # Compute shift s = 2^K mod l
            if l in mod_cache:
                s = mod_cache[l]
            else:
                s = pow(2, K, l)
                mod_cache[l] = s
            # Place each element shifted by s in the cycle
            for j, idx in enumerate(cycle):
                ans[idx] = cycle[(j + s) % l]
    # Convert zero-based to one-based and output
    out = " ".join(str(v+1) for v in ans)
    sys.stdout.write(out)

if __name__ == "__main__":
    main()