class Trie:
    def __init__(self):
        self.root = {}
    
    def insert(self, number):
        node = self.root
        for i in reversed(range(32)):  # Assuming 32-bit integers
            bit = (number >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]
    
    def find_max_xor(self, number):
        if not self.root:
            return 0
        node = self.root
        max_xor = 0
        for i in reversed(range(32)):
            bit = (number >> i) & 1
            desired_bit = 1 - bit
            if desired_bit in node:
                max_xor |= (1 << i)
                node = node[desired_bit]
            else:
                if bit in node:
                    node = node[bit]
                else:
                    break
        return max_xor

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    nums = list(map(int, input[ptr:ptr + n]))
    ptr += n
    q = int(input[ptr])
    ptr += 1
    queries = []
    for _ in range(q):
        l = int(input[ptr])
        r = int(input[ptr + 1])
        queries.append((l, r))
        ptr += 2
    
    # Compute prefix_xor
    n = len(nums)
    prefix_xor = [0] * (n + 1)
    for i in range(n):
        prefix_xor[i + 1] = prefix_xor[i] ^ nums[i]
    
    # Precompute max_xor[l][r]
    max_xor = [[0] * n for _ in range(n)]
    for l in range(n):
        trie = Trie()
        trie.insert(prefix_xor[l])
        current_max = 0
        max_xor[l][l] = 0
        for r in range(l, n - 1):
            current = prefix_xor[r + 1]
            max_val = trie.find_max_xor(current)
            if max_val > current_max:
                current_max = max_val
            max_xor[l][r] = current_max
            trie.insert(current)
    
    # Answer queries
    for a, b in queries:
        current_max = 0
        for l in range(a, b + 1):
            if b < n:
                current_max = max(current_max, max_xor[l][b])
        print(current_max)

if __name__ == "__main__":
    main()