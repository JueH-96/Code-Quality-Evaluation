import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    n, q = map(int, data[0].split())
    s = data[1].strip()
    
    if n == 1:
        output_lines = []
        for i in range(2, 2+q):
            parts = data[i].split()
            if parts[0] == '2':
                output_lines.append("Yes")
        sys.stdout.write("
".join(output_lines))
        return
        
    arr = [0] * (n-1)
    for i in range(n-1):
        if s[i] == s[i+1]:
            arr[i] = 1
        else:
            arr[i] = 0

    class SegmentTree:
        def __init__(self, data):
            self.n = len(data)
            if self.n == 0:
                self.size = 0
                self.tree = []
                return
            self.size = 1
            while self.size < self.n:
                self.size *= 2
            self.tree = [0] * (2 * self.size)
            for i in range(self.n):
                self.tree[self.size + i] = data[i]
            for i in range(self.size-1, 0, -1):
                self.tree[i] = max(self.tree[2*i], self.tree[2*i+1])
        
        def update(self, index, value):
            if self.n == 0:
                return
            pos = self.size + index
            self.tree[pos] = value
            pos //= 2
            while pos:
                self.tree[pos] = max(self.tree[2*pos], self.tree[2*pos+1])
                pos //= 2
                
        def query(self, l, r):
            if l > r or self.n == 0:
                return 0
            l += self.size
            r += self.size
            res = 0
            while l <= r:
                if l % 2 == 1:
                    res = max(res, self.tree[l])
                    l += 1
                if r % 2 == 0:
                    res = max(res, self.tree[r])
                    r -= 1
                l //= 2
                r //= 2
            return res
            
    seg_tree = SegmentTree(arr)
    
    output_lines = []
    for i in range(2, 2+q):
        parts = data[i].split()
        t = parts[0]
        L = int(parts[1])
        R = int(parts[2])
        l0 = L - 1
        r0 = R - 1
        if t == '1':
            if l0 >= 1:
                idx = l0 - 1
                arr[idx] = 1 - arr[idx]
                seg_tree.update(idx, arr[idx])
            if r0 < n-1:
                idx = r0
                arr[idx] = 1 - arr[idx]
                seg_tree.update(idx, arr[idx])
        else:
            length = R - L + 1
            if length == 1:
                output_lines.append("Yes")
            else:
                low = l0
                high = r0 - 1
                res = seg_tree.query(low, high)
                if res == 0:
                    output_lines.append("Yes")
                else:
                    output_lines.append("No")
                    
    sys.stdout.write("
".join(output_lines))

if __name__ == "__main__":
    main()