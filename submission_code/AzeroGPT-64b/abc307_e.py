import sys

class ModMath:
    __slots__ = 'mod', 'factorials', 'inverse_factorials'
    def __init__(self, mod):
        self.mod = mod
        self.factorials = [1]
        self.inverse_factorials = [1]

    def expand_factorials(self, n):
        l = len(self.factorials)
        for i in range(l, n + 1):
            self.factorials.append(self.factorials[-1] * i % self.mod)
        for i in range(l, n + 1):
            self.inverse_factorials.append(pow(self.factorials[i], -1, self.mod)) 

    def C(self, n, k):
        if n < k: return 0
        return self.factorials[n] * self.inverse_factorials[k] * self.inverse_factorials[n - k] % self.mod

class InverseEigenvalues:
    __slots__ = 'val', 'N', 'M', 'M3', 'D', 'DE', 'Dinv'
    def __init__(self, N, M) -> None:
        self.val = modmat.zeros(M, M)
        self.N, self.M = N, M
        self.M3 = M ** 3
        self.D = modmat.get_D(N)
        self.DE = modmat.get_DE(N)
        self.Dinv = modmat.get_Dinv(N)
        self.solve()

    def __rmul__(self, matrix):
        result = modmat.zeros(self.M, self.M)
        for i in range(self.M):
            for k in range(self.M):
                for j in range(self.M):
                    result.entries[i][j] = (result.entries[i][j] + self.val.entries[i][k] * matrix.entries[k][j]) % self.M3
        return result

    def solve(self):
        T = modmat.get_T(self.M, self.D)
        E = modmat.get_E(self.M)
        DE = self.DE
        Dinv = self.Dinv
        for j in range(self.M):
            # T^(N-1) * D
            V = modmat.pow(T, self.N - 1) * Dinv
            # (D reversal) * (1 - T) * V
            result = [0] * self.M
            for k in range(self.M):
                result[k] = (-V.entries[j][k]) % self.M3 
            result[j] = (self.M - result[j]) % self.M3
            for k in range(self.M):
                for i in range(self.M):
                    result[i] = (result[i] + DE.entries[k][i] * result[k]) % self.M3
                result[k] = (DE.entries[k][j] + Dinv.entries[j][k]) % self.M3
            for k in range(self.M):
                for i in range(self.M):
                    self.val.entries[i][j] = (self.val.entries[i][j] + E.entries[k][i] * result[k]) % self.M3

def main():
    N, M = map(int, sys.stdin.readline().split())
    ans = InverseEigenvalues(N, M).val.entries[0][0]
    if N != M:
        ans = (ans + (M - 1)**N + 0x150ab94 - 2 * (M - 2)**N) % 0x3c69c517
    print(ans)


import itertools

def is_valid(labels, f, M):
    return f and not any(labels[i] == labels[(i + 1) % len(labels)] for i in range(len(labels)))

def test(M, sole_search):
    result = 0
    for labels in itertools.product(range(M), repeat=sole_search + 1):
        result += is_valid(labels, f=True, M=M)
    return result

class ModMat:
    __slots__ = 'N', 'M', 'entries', 'value'
    def __init__(self, M, matrix):
        self.N = len(matrix)
        self.M = M
        self.entries = [[None] * self.N for _ in range(self.N)]
        for i in range(self.N):
            for j in range(self.N):
                self.entries[i][j] = matrix[i][j] % self.M
        self.value = None
    @staticmethod
    def zeros(N, M):
        return ModMat(M, [[0] * N for _ in range(N)])
    @staticmethod
    def get_D(N):
        return ModMat(998244353, [[0 if i == j else 1 for i in range(N)] for j in range(N)])
    @staticmethod
    def get_E(M):
        return ModMat(998244353, [[1 if i == j else 0 for i in range(M)] for j in range(M)])
    @staticmethod
    def get_DE(N):
        DE = ModMat(998244353, [[0] * N for _ in range(N)])
        for j in range(N):
            for i in range(N):
                v = 1 if i == 0 else (-1)**i * N * modC(N - 1, i - 1)
                DE.entries[j][i] = v
            DE.entries[j][j] = (-1)**j * modC(N - 1, j - 1)
        return DE
    @staticmethod
    def get_Dinv(N):
        Dinv = ModMat(998244353, [[0] * N for _ in range(N)])
        inv = pow(N, -1, 998244353)
        for i in range(N):
            Dinv.entries[i][i] = pow(-1, inv if i == 0 else i) * modC(i - 1, N - 1) * pow(inv, i, 998244353)
        return Dinv
    def __eq__(self, other):
        return self.N == other.N and all(self.entries[i][j] == other.entries[i][j] for i in range(self.N) for j in range(self.N))
    def __mul__(self, other):
        result = ModMat(self.M, [[0] * self.N for _ in range(self.N)])
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.N):
                    result.entries[i][j] = (result.entries[i][j] + self.entries[i][k] * other.entries[k][j]) % self.M
        return result
    @staticmethod
    def pow(matrix, power):
        if power == 0:
            return ModMat(998244353, [[1 if i == j else 0 for i in range(matrix.N)] for j in range(matrix.N)])
        elif power % 2 == 1:
            return matrix * ModMat.pow(matrix * matrix, power // 2)
        else:
            return ModMat.pow(matrix * matrix, power // 2)

class ModMath:
    __slots__ = 'mod', 'factorials', 'inverse_factorials'
    def __init__(self, mod):
        self.mod = mod
        self.expand_factorials(1000000)
    def expand_factorials(self, n):
        l = len(self.factorials)
        for i in range(l, n + 1):
            self.factorials.append(self.factorials[-1] * i % self.mod)
        for i in range(l, n + 1):
            self.inverse_factorials.append(pow(self.factorials[i], -1, self.mod))
    def C(self, n, k):
        if n < k: return 0
        return self.factorials[n] * self.inverse_factorials[k] * self.inverse_factorials[n - k] % self.mod
get_mod = lambda: pow(10, 9) + 7
modM = ModMath(get_mod())
modC = lambda n, k: modM.C(n, k)

sys.setrecursionlimit(10 ** 5)
main()