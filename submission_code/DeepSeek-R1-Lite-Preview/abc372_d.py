def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    H = list(map(int, N_and_rest[1:N+1]))
    
    c = [0] * N
    stack = []
    
    for j in range(N):
        while stack and H[j] > H[stack[-1]]:
            i = stack.pop()
            c[i] = j - i
        stack.append(j)
    
    while stack:
        i = stack.pop()
        c[i] = N - i - 1
    
    print(' '.join(map(str, c)))

if __name__ == '__main__':
    main()