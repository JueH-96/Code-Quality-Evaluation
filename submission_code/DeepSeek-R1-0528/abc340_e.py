class Fenw:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def add(self, index, delta):
        i = index + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def prefix_sum(self, index):
        i = index + 1
        s = 0
        while i:
            s += self.tree[i]
            i -= i & -i
        return s

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2 + n]))
    B = list(map(int, data[2 + n:2 + n + m]))
    
    fenw = Fenw(n)
    global_base = 0
    
    for b in B:
        current_val = A[b] + global_base + fenw.prefix_sum(b)
        if current_val == 0:
            base_current = 0
            rem = 0
        else:
            base_current = current_val // n
            rem = current_val % n
        
        fenw.add(b, -current_val)
        if b + 1 < n:
            fenw.add(b + 1, current_val)
        
        global_base += base_current
        
        if rem > 0:
            start = (b + 1) % n
            if start + rem <= n:
                fenw.add(start, 1)
                end = start + rem
                if end < n:
                    fenw.add(end, -1)
            else:
                fenw.add(start, 1)
                remaining_length = rem - (n - start)
                r = remaining_length - 1
                fenw.add(0, 1)
                if r + 1 < n:
                    fenw.add(r + 1, -1)
    
    result = []
    for i in range(n):
        total = A[i] + global_base + fenw.prefix_sum(i)
        result.append(str(total))
    
    print(" ".join(result))

if __name__ == "__main__":
    main()