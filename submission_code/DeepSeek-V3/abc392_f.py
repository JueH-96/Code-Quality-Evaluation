import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)
    
    def update(self, index, delta=1):
        while index <= self.n:
            self.tree[index] += delta
            index += index & -index
    
    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res
    
    def find_kth(self, k):
        index = 0
        bit_mask = 1 << (self.n.bit_length())
        while bit_mask != 0:
            temp = index + bit_mask
            if temp <= self.n and self.tree[temp] < k:
                index = temp
                k -= self.tree[index]
            bit_mask >>= 1
        return index + 1

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    P = list(map(int, input[ptr:ptr+N]))
    
    ft = FenwickTree(N)
    # Initialize the Fenwick tree with 1's (all positions are available)
    for i in range(1, N+1):
        ft.update(i, 1)
    
    res = [0] * N
    # Process in reverse order
    for i in range(N, 0, -1):
        p = P[i-1]
        # Find the position where there are exactly p positions before it
        low = 1
        high = N
        pos = 0
        # Using binary search to find the kth position
        # The Fenwick tree's find_kth method can be used here
        k = p
        pos = ft.find_kth(k)
        res[pos - 1] = i
        ft.update(pos, -1)
    
    print(' '.join(map(str, res)))

if __name__ == '__main__':
    main()