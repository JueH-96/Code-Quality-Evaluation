class BIT:
    def __init__(self, size):
        self.N = size + 2  # To handle 1-based indexing
        self.tree = [0] * (self.N)
    
    def update(self, index, value=1):
        while index < self.N:
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
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    chords = []
    index = 1
    for _ in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        index += 2
        p = min(A, B)
        q = max(A, B)
        chords.append((p, q))
    
    # Sort chords by p
    chords.sort()
    
    # Initialize BIT
    bit = BIT(2 * N)
    
    # Iterate through chords in reverse order
    for i in range(N - 1, -1, -1):
        p, q = chords[i]
        # Query for number of q's > current q
        # Since q ranges from 1 to 2N, number of q's greater than q is bit.query(2N) - bit.query(q)
        total = bit.query(2 * N)
        less_or_equal = bit.query(q)
        greater = total - less_or_equal
        if greater > 0:
            print("Yes")
            return
        # Update BIT with current q
        bit.update(q)
    
    print("No")

if __name__ == "__main__":
    main()