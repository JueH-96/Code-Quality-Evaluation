import bisect

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.count = [0] * (self.size + 2)
        self.sum_val = [0] * (self.size + 2)
    
    def update(self, index, value):
        while index <= self.size:
            self.count[index] += 1
            self.sum_val[index] += value
            index += index & -index
    
    def query(self, index):
        cnt = 0
        sm = 0
        while index > 0:
            cnt += self.count[index]
            sm += self.sum_val[index]
            index -= index & -index
        return (sm, cnt)

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    A = list(map(int, input[1:n+1]))
    sorted_A = sorted(A)
    sorted_unique_A = []
    prev = None
    for num in sorted_A:
        if num != prev:
            sorted_unique_A.append(num)
            prev = num
    m = len(sorted_unique_A)
    ft = FenwickTree(m)
    total = 0
    for a in reversed(A):
        original_pos = bisect.bisect_left(sorted_unique_A, a)
        rank = original_pos + 1
        sum_le, count_le = ft.query(rank)
        sum_total, count_total = ft.query(m)
        sum_gt = sum_total - sum_le
        count_gt = count_total - count_le
        total += sum_gt - a * count_gt
        ft.update(rank, a)
    print(total)

if __name__ == "__main__":
    main()