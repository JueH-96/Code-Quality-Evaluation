import sys

class Fenw:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)
    
    def update(self, index, delta):
        i = index + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i
            
    def query_prefix(self, index):
        if index < 0:
            return 0
        s = 0
        i = index + 1
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s
            
    def range_query(self, l, r):
        if l > r:
            return 0
        if l == 0:
            return self.query_prefix(r)
        else:
            return self.query_prefix(r) - self.query_prefix(l - 1)

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n, q = map(int, data[0].split())
    s = data[1].strip()
    
    if n == 1:
        output_lines = []
        for i in range(2, 2 + q):
            line = data[i].split()
            if line[0] == '2':
                output_lines.append("Yes")
        print("
".join(output_lines))
        return
        
    adj = []
    for i in range(n - 1):
        if s[i] != s[i + 1]:
            adj.append(1)
        else:
            adj.append(0)
            
    fenw = Fenw(n - 1)
    for i in range(n - 1):
        fenw.update(i, adj[i])
        
    output_lines = []
    for i in range(2, 2 + q):
        parts = data[i].split()
        t = int(parts[0])
        L = int(parts[1])
        R = int(parts[2])
        l0 = L - 1
        r0 = R - 1
        
        if t == 1:
            if l0 > 0:
                idx = l0 - 1
                delta = 1 - 2 * adj[idx]
                fenw.update(idx, delta)
                adj[idx] = 1 - adj[idx]
            if r0 < n - 1:
                idx = r0
                delta = 1 - 2 * adj[idx]
                fenw.update(idx, delta)
                adj[idx] = 1 - adj[idx]
        else:
            if l0 == r0:
                output_lines.append("Yes")
            else:
                total = fenw.range_query(l0, r0 - 1)
                length_segment = r0 - l0
                if total == length_segment:
                    output_lines.append("Yes")
                else:
                    output_lines.append("No")
                    
    print("
".join(output_lines))

if __name__ == "__main__":
    main()