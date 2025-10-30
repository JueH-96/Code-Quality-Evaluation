class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def update(self, idx, delta):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    
    fenwick = FenwickTree(M)
    fenwick.update(0 + 1, 1)  # S[0] = 0
    current_sum = 0
    total_sum = 0
    answer = 0
    
    for i in range(1, N + 1):
        current_sum = (current_sum + A[i-1]) % M
        x = current_sum
        count_gt = i - fenwick.query(x + 1)
        contribution = x * i - total_sum + M * count_gt
        answer += contribution
        fenwick.update(x + 1, 1)
        total_sum += x
    
    print(answer)

if __name__ == "__main__":
    main()