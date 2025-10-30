import sys

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 2)  # 1-based indexing

    def update(self, index, delta):
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    
    original_pos = [0] * (N + 1)  # 1-based for values 1..N
    for i in range(N):
        original_pos[P[i]] = i  # 0-based
    
    # Compute inv_count_for_position
    inv_count_for_position = [0] * N
    fenwick = FenwickTree(N)
    for i in range(N-1, -1, -1):
        current_value = P[i]
        cnt = fenwick.query(current_value - 1)
        inv_count_for_position[i] = cnt
        fenwick.update(current_value, 1)
    
    total_cost = 0
    for x in range(1, N+1):
        i_0based = original_pos[x]
        ic = inv_count_for_position[i_0based]
        original_pos_1based = i_0based + 1
        current_pos_1based = original_pos_1based + ic
        steps = current_pos_1based - x
        if steps > 0:
            a = x
            b = current_pos_1based - 1
            sum_range = (a + b) * steps // 2
            total_cost += sum_range
    print(total_cost)

if __name__ == "__main__":
    main()