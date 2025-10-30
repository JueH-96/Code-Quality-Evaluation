import sys
input = sys.stdin.readline

class Graph:
    def __init__(self, N, mod=998244353):
        self.fact = [1, 1]
        self.inv   = [0, 1]
        self.finv = [1, 1]
        self.mod   = mod

        for i in range(2, N + 1):
            self.fact.append((self.fact[-1] * i) % self.mod)
            self.inv.append((-self.inv[self.mod % i] * (self.mod//i)) % self.mod)
            self.finv.append((self.finv[-1] * self.inv[-1]) % self.mod)


    def perm(self, n, r, p):
        if r < 0 or r > n:
            return 0
        else:
            return self.fact[n] * self.finv[r] * self.finv[n - r] % self.mod

    def comb(self, n, r, p):
        if r < 0 or r > n:
            return 0
        else:
            return self.fact[n] * self.finv[r] * self.finv[n - r] % self.mod

    def hock(self, n, r, p):
        if r < 0 or r > n:
            return 0
        else:
            return self.fact[n + r - 1] * self.finv[r] * self.finv[n - 1] % self.mod

    def fact_inv(self, i):
        return self.finv[i]

def solve(N, A, B):
    g = Graph(2 * N)

    used_verteces = [False] * (2 * N)
    for e1 in A:
        used_verteces[e1 - 1] = True
    for q in B:
        if q >= 1:
            used_verteces[q - 1] = True

    vertices_1 = []
    for i in range(2 * N):
        if not used_verteces[i]:
            vertices_1.append(ut(i))

    assert len(vertices_1) % 2 == 0
    M = len(vertices_1) // 2

    degree_in = [0] * (2 * N)

    for i, e in enumerate(A):
        v = e - 1
        if B[i] >= 1:
            u = B[i] - 1
        else:
            vertices_1.pop()
            u = vertices_1.pop()
            vertices_1.append(u)

        degree_in[v] += 1
        degree_in[u] += 1

    edges_not_selected = 0
    verteces_not_selected = [False] * (2 * N)
    allowed_deg_one = 0
    verteces_undirected = []

    for i in range(2 * N):
        if not verteces_not_selected[i]:
            if degree_in[i] == 0:
                verteces_undirected.append(i)
            elif degree_in[i] == 1:
                allowed_deg_one += 1
            else:
                assert degree_in[i] == 2
                edges_not_selected += 1
            verteces_not_selected[i] = True

    if len(vertices_1) != allowed_deg_one:
        return 0

    edges_checking = []
    allowed_deg_two = M - len(vertices_1)

    for i in verteces_undirected:
        if degree_in[i] == 0:
            verteces_undirected.pop()
            v = vertices_1.pop()
            vertices_1.append(v)
            deg_g = 0
            while True:
                if deg_g == 2:
                    edges_not_selected += 1
                    edges_checking.append([u, v])
                    break
                if deg_g == 1:
                    if allowed_deg_one == 0:
                        v = ut(1)
                        while True:
                            v = vertices_1.pop()
                            if degree_in[v] == 0:
                                break
                            vertices_1.append(v)
                        vertices_1.append(v)
                        allowed_deg_one -= 1
                    if allowed_deg_two == 0:
                        u = ut(1)
                        while True:
                            u = vertices_1.pop()
                            if degree_in[u] == 0:
                                break
                            vertices_1.append(u)
                        vertices_1.append(u)
                        allowed_deg_one -= 1
                    else:
                        u = vertices_1.pop()
                        vertices_1.append(u)
                if deg_g == 0:
                    allowed_deg_two -= 1
                    u = vertices_1.pop()
                    vertices_1.append(u)
                if allowed_deg_one < 0 or allowed_deg_two < 0:
                    return 0
                deg_g += 1

    for u, v in edges_checking:
        if degree_in[u] == 1 and degree_in[v] == 1:
            edges_not_selected += 1
        elif degree_in[u] == 1 or degree_in[v] == 1:
            v = vertices_1.pop()
            vertices_1.append(v)
        else:
            u = vertices_1.pop()
            v = vertices_1.pop()
            vertices_1.append(v)
            vertices_1.append(u)

    assert len(vertices_1) == 0
    if allowed_deg_one <= 0 or allowed_deg_two <= 0:
        return 0

    ans = 1
    for e in edges_not_selected:
        ans *= 2
        ans %= 998244353
    ans *= g.perm(allowed_deg_two, allowed_deg_one, 998244353)
    ans %= 998244353

    return ans

N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

print(solve(N, A, B))