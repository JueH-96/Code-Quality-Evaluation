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
    N = int(input[0])
    P = list(map(int, input[1:N+1]))
    
    original_pos = [0] * (N + 1)
    for i in range(N):
        x = P[i]
        original_pos[x] = i + 1  # 1-based position

    left = [0] * (N + 1)
    ft = FenwickTree(N)
    for i in range(N):
        x = P[i]
        sum_high = ft.query(N) - ft.query(x)
        left[x] = sum_high
        ft.update(x, 1)
    
    total_cost = 0
    for x in range(N, 0, -1):
        pos = original_pos[x]
        current_pos = pos - left[x]
        if current_pos < x:
            a = current_pos
            b = x - 1
            if a <= b:
                cost = (a + b) * (b - a + 1) // 2
                total_cost += cost
        elif current_pos > x:
            a = x
            b = current_pos - 1
            cost = (a + b) * (b - a + 1) // 2
            total_cost += cost
    
    print(total_cost)

if __name__ == '__main__':
    main()