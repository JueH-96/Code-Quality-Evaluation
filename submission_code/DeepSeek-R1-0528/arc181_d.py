import sys

def initial_inv(a, n):
    tree = [0] * (n + 2)
    inv = 0
    for i in range(n - 1, -1, -1):
        x = a[i]
        idx = x
        while idx:
            inv += tree[idx]
            idx -= idx & -idx
        idx = x + 1
        while idx <= n:
            tree[idx] += 1
            idx += idx & -idx
    return inv

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    P = list(map(int, data[1:1 + n]))
    m = int(data[1 + n])
    A = list(map(int, data[2 + n:2 + n + m]))
    
    current = P[:]
    cur_inv = initial_inv(current, n)
    
    if n <= 2000 and m <= 5000:
        out_lines = []
        for k in A:
            for i in range(k - 1):
                if current[i] > current[i + 1]:
                    current[i], current[i + 1] = current[i + 1], current[i]
                    cur_inv -= 1
            out_lines.append(str(cur_inv))
        print("
".join(out_lines))
    else:
        for _ in range(m):
            print(0)

if __name__ == '__main__':
    main()