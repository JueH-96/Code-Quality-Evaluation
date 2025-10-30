def next_collection(q, r, d):
    if d <= r:
        return r
    else:
        return r + ((d - r + q - 1) // q) * q

def main():
    import sys
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    garbage = []
    for _ in range(N):
        q = int(data[ptr])
        r = int(data[ptr+1])
        garbage.append((q, r))
        ptr += 2
    Q = int(data[ptr])
    ptr += 1
    for _ in range(Q):
        t = int(data[ptr]) - 1  # 0-based index
        d = int(data[ptr+1])
        q, r = garbage[t]
        x = next_collection(q, r, d)
        print(x)
        ptr += 2

if __name__ == '__main__':
    main()