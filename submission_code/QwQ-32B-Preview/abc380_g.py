class FenwickTree:
    def __init__(self, size):
        self.N = size
        self.tree = [0] * (size + 1)
    
    def update(self, index, delta):
        while index <= self.N:
            self.tree[index] += delta
            index += index & -index
    
    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

def inversion_count(P, N):
    inv = 0
    ft = FenwickTree(N)
    for y in range(N-1, -1, -1):
        inv = (inv + ft.query(P[y]-1)) % 998244353
        ft.update(P[y], 1)
    return inv

def sum_y(P, N):
    sum_y = 0
    ft = FenwickTree(N)
    for y in range(N-1, -1, -1):
        sum_y = (sum_y + (ft.query(N) - ft.query(P[y]-1))) % 998244353
        ft.update(P[y], y+1)
    return sum_y

def sum_x(P, N):
    sum_x = 0
    ft = FenwickTree(N)
    for x in range(N):
        sum_x = (sum_x + ft.query(P[x]-1)) % 998244353
        ft.update(P[x], x+1)
    return sum_x

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:2+N]))
    
    Inv_P = inversion_count(P, N)
    sum_y_val = sum_y(P, N)
    sum_x_val = sum_x(P, N)
    
    M = N - K + 1
    if M == 0:
        print(0)
        return
    
    sum_inversions_K_minus_y_plus_x = (Inv_P * K - (sum_y_val - sum_x_val)) % 998244353
    average_Inv_B = (sum_inversions_K_minus_y_plus_x * pow(M, 998244353-2, 998244353)) % 998244353
    expected_Inv = (Inv_P - average_Inv_B + (K * (K-1) // 4) % 998244353) % 998244353
    print(expected_Inv)

if __name__ == "__main__":
    main()