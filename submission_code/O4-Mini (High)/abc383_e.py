import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, M, K = map(int, input().split())
    parent = list(range(N+1))
    size = [1] * (N+1)
    # comp_count[i] > 0 means that component root i has that many A's unmatched,
    # comp_count[i] < 0 means that it has -comp_count[i] B's unmatched.
    comp_count = [0] * (N+1)

    edges = []
    for _ in range(M):
        u, v, w = map(int, input().split())
        edges.append((w, u, v))

    # Read A's: increment count
    for a in map(int, input().split()):
        comp_count[a] += 1
    # Read B's: decrement count
    for b in map(int, input().split()):
        comp_count[b] -= 1

    # Sort edges by weight ascending
    edges.sort(key=lambda x: x[0])

    # DSU find with path compression
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    total_cost = 0

    # Process edges in increasing weight order
    for w, u, v in edges:
        ru = find(u)
        rv = find(v)
        if ru == rv:
            continue
        cu = comp_count[ru]
        cv = comp_count[rv]
        # If one component has A's (positive) and the other has B's (negative),
        # we can match as many as min(|cu|,|cv|) at cost w each.
        if cu * cv < 0:
            # number of matches
            t = min(abs(cu), abs(cv))
            total_cost += w * t
        # New unmatched count after merging
        new_count = cu + cv
        # Union by size: attach smaller tree under larger
        if size[ru] < size[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        size[ru] += size[rv]
        comp_count[ru] = new_count

    print(total_cost)

if __name__ == "__main__":
    main()