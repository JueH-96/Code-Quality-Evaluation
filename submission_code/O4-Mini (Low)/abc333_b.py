def main():
    import sys

    # Read the two segments from stdin
    s = sys.stdin.readline().strip()
    t = sys.stdin.readline().strip()

    # Map vertex labels to indices 0..4
    idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    def pentagon_distance(u, v):
        # Compute the minimal distance (in edges) between two vertices on a cycle of length 5
        duv = abs(idx[u] - idx[v])
        return min(duv, 5 - duv)

    # Compute the two distances
    d1 = pentagon_distance(s[0], s[1])
    d2 = pentagon_distance(t[0], t[1])

    # Compare and output result
    print("Yes" if d1 == d2 else "No")


if __name__ == "__main__":
    main()