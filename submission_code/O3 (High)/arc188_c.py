import sys

sys.setrecursionlimit(1 << 25)


class DSUParity:
    """
    Disjoint Set Union that keeps the xor (parity) between a node and its root.
    For every edge (u, v, w) we want value(u) XOR value(v) == w (w is 0/1).
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.parity = [0] * n        # parity[node] = value(node) XOR value(parent[node])
        self.rank = [0] * n          # for union by rank

    def find(self, x: int):
        """
        Returns (root, p) where p = value(x) XOR value(root).
        Path compression is applied while maintaining parity.
        """
        if self.parent[x] != x:
            r, p = self.find(self.parent[x])
            self.parity[x] ^= p
            self.parent[x] = r
        return self.parent[x], self.parity[x]

    def unite(self, x: int, y: int, w: int) -> bool:
        """
        Merges the sets of x and y enforcing value(x) XOR value(y) == w.
        Returns True if merged successfully, False if a contradiction is found.
        """
        rx, px = self.find(x)
        ry, py = self.find(y)

        # If already in the same set, check consistency
        if rx == ry:
            return (px ^ py) == w

        # Union by rank – attach smaller tree under larger
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
            px, py = py, px

        # Make ry child of rx
        self.parent[ry] = rx
        # Set parity[ry] so that the required relation holds
        # value(ry) XOR value(rx)  should equal  px XOR py XOR w
        self.parity[ry] = px ^ py ^ w

        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

        return True


def main() -> None:
    input_data = sys.stdin.buffer.readline
    N, M = map(int, input_data().split())

    # We create two variables for every villager i (0-based):
    #   X_i : honesty of villager i        →  index i
    #   Z_i : X_i XOR Conf_i (defined in analysis) → index N + i
    total_vars = 2 * N
    dsu = DSUParity(total_vars)

    ok = True
    for _ in range(M):
        A, B, C = map(int, input_data().split())
        u = N + (A - 1)   # Z_A
        v = (B - 1)       # X_B
        if not dsu.unite(u, v, C):
            ok = False
            break

    if not ok:
        print(-1)
        return

    # Build one concrete solution: choose every root value = 0
    res = []
    for i in range(N):
        _, px = dsu.find(i)         # value of X_i  (mod 2)
        _, pz = dsu.find(N + i)     # value of Z_i  (mod 2)
        conf = px ^ pz              # Conf_i = X_i XOR Z_i
        res.append('1' if conf else '0')

    print(''.join(res))


if __name__ == "__main__":
    main()