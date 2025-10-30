import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    a = list(map(int, data[1:n+1]))
    
    # Coordinate compression
    sorted_unique = sorted(list(set(a)))
    max_rank = len(sorted_unique)
    
    class FenwickTree:
        def __init__(self, size):
            self.size = size
            self.tree = [0] * (self.size + 2)  # Adding +2 to avoid issues with zero
        
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
    
    count_tree = FenwickTree(max_rank)
    sum_tree = FenwickTree(max_rank)
    
    total = 0
    
    for x in a:
        r = bisect.bisect_left(sorted_unique, x) + 1  # 1-based rank
        count = count_tree.query(r - 1)
        sum_val = sum_tree.query(r - 1)
        contrib = x * count - sum_val
        total += contrib
        
        count_tree.update(r, 1)
        sum_tree.update(r, x)
    
    print(total)

if __name__ == '__main__':
    main()