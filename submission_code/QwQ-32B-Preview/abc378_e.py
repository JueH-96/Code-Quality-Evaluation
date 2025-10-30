class Fenwick:
    def __init__(self, size):
        self.N = size
        self.tree = [0] * (self.N + 1)
    
    def update(self, pos, val):
        while pos <= self.N:
            self.tree[pos] += val
            pos += pos & -pos
    
    def query(self, pos):
        res = 0
        while pos > 0:
            res += self.tree[pos]
            pos -= pos & -pos
        return res

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    S = [0] * (N + 1)
    for i in range(1, N + 1):
        S[i] = S[i - 1] + A[i - 1]
    
    P = [S[i] % M for i in range(N + 1)]
    
    Q = [0] * (N + 1)
    for j in range(1, N + 1):
        Q[j] = Q[j - 1] + P[j]
    
    total_sum_part1 = 0
    for j in range(1, N + 1):
        total_sum_part1 += j * P[j] - Q[j - 1]
    
    unique_P = sorted(set(P))
    rank = {P: i + 1 for i, P in enumerate(unique_P)}
    M_fenwick = len(unique_P)
    fen = Fenwick(M_fenwick)
    inv_count = 0
    for j in range(N, 0, -1):
        rank_pj = rank[P[j]]
        inv_count += fen.query(M_fenwick) - fen.query(rank_pj)
        fen.update(rank_pj, 1)
    
    answer = total_sum_part1 + M * inv_count
    print(answer)

if __name__ == "__main__":
    main()