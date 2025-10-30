import bisect

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
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0

    N = int(data[ptr])
    ptr +=1
    K = int(data[ptr])
    ptr +=1
    Q = int(data[ptr])
    ptr +=1

    A = [0] * N

    all_values = set()
    all_values.add(0)
    for _ in range(Q):
        X = int(data[ptr])-1
        ptr +=1
        Y = int(data[ptr])
        ptr +=1
        all_values.add(Y)
    sorted_values = sorted(all_values)
    compressed = {v:i+1 for i, v in enumerate(sorted_values)}  # 1-based indexing
    max_rank = len(sorted_values)

    ft_count = FenwickTree(max_rank)
    ft_sum = FenwickTree(max_rank)

    A_compressed = [compressed[0]] * N

    for _ in range(Q):
        X = int(data[ptr])-1
        ptr +=1
        Y = int(data[ptr])
        ptr +=1

        old_val = A[X]
        new_val = Y

        old_rank = compressed[old_val]
        new_rank = compressed[new_val]

        ft_count.update(old_rank, -1)
        ft_sum.update(old_rank, -old_val)
        ft_count.update(new_rank, 1)
        ft_sum.update(new_rank, new_val)

        A[X] = new_val

        def find_kth():
            left = 1
            right = max_rank
            best = -1
            while left <= right:
                mid = (left + right) // 2
                cnt = ft_count.query(mid)
                if cnt >= K:
                    best = mid
                    right = mid -1
                else:
                    left = mid +1
            return best

        v = find_kth()
        sum_val = ft_sum.query(v)
        print(sum_val)

if __name__ == '__main__':
    main()