# YOUR CODE HERE
import sys

class BIT:
    def __init__(self, size):
        self.N = size
        self.tree = [0] * (self.N + 2)

    def update(self, index):
        while index <= self.N:
            self.tree[index] += 1
            index += index & -index

    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    chords = []
    for i in range(N):
        A = int(data[1 + 2*i])
        B = int(data[2 + 2*i])
        if A > B:
            A, B = B, A
        chords.append((A, B))
    chords.sort()
    bit = BIT(2*N)
    for A, B in chords:
        count = bit.query(2*N) - bit.query(B)
        if count > 0:
            print("Yes")
            return
        bit.update(B)
    print("No")

if __name__ == "__main__":
    main()