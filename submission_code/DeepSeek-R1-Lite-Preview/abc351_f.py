import sys

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 1)
    
    def update(self, index, value):
        while index <= self.size:
            self.tree[index] += value
            index += index & -index
    
    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    sorted_unique_A = sorted(set(A))
    rank_dict = {value: rank for rank, value in enumerate(sorted_unique_A, start=1)}
    max_rank = len(sorted_unique_A)
    
    sum_fenwick = FenwickTree(max_rank)
    count_fenwick = FenwickTree(max_rank)
    
    total_sum = 0
    for i in range(N-1, -1, -1):
        rank_i = rank_dict[A[i]]
        sum_Aj = sum_fenwick.query(max_rank) - sum_fenwick.query(rank_i)
        count_j = count_fenwick.query(max_rank) - count_fenwick.query(rank_i)
        total_sum += sum_Aj - count_j * A[i]
        sum_fenwick.update(rank_i, A[i])
        count_fenwick.update(rank_i, 1)
    
    print(total_sum)

if __name__ == '__main__':
    main()