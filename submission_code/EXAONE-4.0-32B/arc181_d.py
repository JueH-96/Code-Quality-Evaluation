import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    P = list(map(int, data[1:1+n]))
    m = int(data[1+n])
    A = list(map(int, data[1+n+1:1+n+1+m]))
    
    size = 200000
    fenw = [0] * (size + 1)
    
    def update(idx, delta):
        while idx <= size:
            fenw[idx] += delta
            idx += idx & -idx
            
    def query(idx):
        res = 0
        while idx:
            res += fenw[idx]
            idx -= idx & -idx
        return res
        
    inv = 0
    for i in range(n - 1, -1, -1):
        num = P[i]
        inv += query(num - 1)
        update(num, 1)
        
    results = []
    for a in A:
        for i in range(a - 1):
            if P[i] > P[i + 1]:
                P[i], P[i + 1] = P[i + 1], P[i]
                inv -= 1
        results.append(str(inv))
    
    print("
".join(results))

if __name__ == "__main__":
    main()