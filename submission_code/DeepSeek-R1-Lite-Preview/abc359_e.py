def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    H = list(map(int, data[1:N+1]))
    
    t = [0] * (N + 1)
    for i in range(1, N + 1):
        t[i] = t[i - 1] + H[i - 1] + 1
        if i > 1:
            t[i] = max(t[i], t[i - 1] + 1)
    
    print(' '.join(map(str, t[1:])))

if __name__ == "__main__":
    main()