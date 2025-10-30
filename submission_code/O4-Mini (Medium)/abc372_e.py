import sys
import threading

def main():
    import sys
    data = sys.stdin
    readline = data.readline

    N_Q = readline().split()
    if not N_Q:
        return
    N, Q = map(int, N_Q)
    parent = list(range(N+1))
    size = [1] * (N+1)
    # For each root, store up to 10 largest elements in its component, sorted descending
    tops = [[i] for i in range(N+1)]

    def find(x):
        # iterative path compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(u, v):
        ru = find(u)
        rv = find(v)
        if ru == rv:
            return
        # attach smaller component to larger
        if size[ru] > size[rv]:
            ru, rv = rv, ru
        # now size[ru] <= size[rv]
        parent[ru] = rv
        size[rv] += size[ru]
        # merge tops[ru] and tops[rv] into new tops[rv]
        a = tops[ru]
        b = tops[rv]
        merged = []
        i = j = 0
        # both a and b are sorted descending
        while len(merged) < 10 and i < len(a) and j < len(b):
            if a[i] > b[j]:
                merged.append(a[i])
                i += 1
            else:
                merged.append(b[j])
                j += 1
        # if still space, take from a
        while len(merged) < 10 and i < len(a):
            merged.append(a[i])
            i += 1
        # then from b
        while len(merged) < 10 and j < len(b):
            merged.append(b[j])
            j += 1
        tops[rv] = merged
        # optional: free memory
        # tops[ru] = []

    output = []
    for _ in range(Q):
        parts = readline().split()
        if not parts:
            continue
        t = int(parts[0])
        if t == 1:
            u = int(parts[1])
            v = int(parts[2])
            union(u, v)
        else:  # t == 2
            v = int(parts[1])
            k = int(parts[2])
            rv = find(v)
            if len(tops[rv]) >= k:
                output.append(str(tops[rv][k-1]))
            else:
                output.append(str(-1))

    sys.stdout.write("
".join(output))

if __name__ == "__main__":
    main()