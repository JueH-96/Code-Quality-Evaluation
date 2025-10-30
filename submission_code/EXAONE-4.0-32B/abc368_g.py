import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    A = [0] * (n+1)
    for i in range(1, n+1):
        A[i] = int(next(it))
    B = [0] * (n+1)
    for i in range(1, n+1):
        B[i] = int(next(it))
    
    q = int(next(it))
    out_lines = []
    for _ in range(q):
        t = next(it)
        if t == '1':
            i = int(next(it))
            x = int(next(it))
            A[i] = x
        elif t == '2':
            i = int(next(it))
            x = int(next(it))
            B[i] = x
        else:
            l = int(next(it))
            r = int(next(it))
            v = 0
            for idx in range(l, r+1):
                add_val = v + A[idx]
                mul_val = v * B[idx]
                if add_val > mul_val:
                    v = add_val
                else:
                    v = mul_val
            out_lines.append(str(v))
    
    print("
".join(out_lines))

if __name__ == "__main__":
    main()