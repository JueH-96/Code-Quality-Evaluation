from typing import List

class FenwickTree:
    def __init__(self):
        self.tree = {}
    
    def update(self, index: int, value: int):
        while index > 0:
            if index in self.tree:
                self.tree[index] += value
            else:
                self.tree[index] = value
            index -= index & -index
    
    def query(self, index: int) -> int:
        res = 0
        while index > 0:
            if index in self.tree:
                res += self.tree[index]
            index -= index & -index
        return res

def calculate_sum(A: List[int]) -> int:
    FT_count = FenwickTree()
    FT_sum = FenwickTree()
    total_sum = 0
    for a in A:
        # Query for elements less than a
        count_j = FT_count.query(a)
        sum_less_j = FT_sum.query(a)
        total_sum += count_j * a - sum_less_j
        # Update the Fenwick Trees
        FT_count.update(a + 1, 1)
        FT_sum.update(a + 1, a)
    return total_sum

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    print(calculate_sum(A))

if __name__ == "__main__":
    main()