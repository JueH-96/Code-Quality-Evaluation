def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    q = [0] * (N + 1)
    r = [0] * (N + 1)
    for i in range(1, N + 1):
        q[i] = int(data[idx])
        r[i] = int(data[idx + 1])
        idx += 2
    Q = int(data[idx])
    idx += 1
    for _ in range(Q):
        t = int(data[idx])
        d = int(data[idx + 1])
        idx += 2
        x = d + ((r[t] - d) % q[t])
        print(x)

if __name__ == "__main__":
    main()