import bisect

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

    def update(self, index, delta):
        while index <= self.n:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    A = list(map(int, input[1:n+1]))
    
    sorted_unique = sorted(set(A))
    m = len(sorted_unique)
    
    sum_tree = FenwickTree(m)
    count_tree = FenwickTree(m)
    
    total = 0
    for x in A:
        r = bisect.bisect_left(sorted_unique, x) + 1
        sum_prev = sum_tree.query(r - 1)
        count_prev = count_tree.query(r - 1)
        total += x * count_prev - sum_prev
        sum_tree.update(r, x)
        count_tree.update(r, 1)
    
    print(total)

if __name__ == "__main__":
    main()