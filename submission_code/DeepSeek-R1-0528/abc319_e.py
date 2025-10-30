import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr]); ptr += 1
    X = int(data[ptr]); ptr += 1
    Y = int(data[ptr]); ptr += 1
    segments = []
    for _ in range(N-1):
        p = int(data[ptr]); t_val = int(data[ptr+1]); ptr += 2
        segments.append((p, t_val))
    
    Q = int(data[ptr]); ptr += 1
    queries = list(map(int, data[ptr:ptr+Q]))
    
    L = 840
    A = list(range(L))
    
    for p, t_val in segments:
        for r in range(L):
            cur = A[r]
            w = (-cur) % p
            A[r] = cur + t_val + w
    
    base = [A[r] - r for r in range(L)]
    
    out_lines = []
    for q in queries:
        t0 = q + X
        r = t0 % L
        ans = t0 + base[r] + Y
        out_lines.append(str(ans))
    
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()