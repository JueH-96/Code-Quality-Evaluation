def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    H = list(map(int, data[1:N+1]))
    
    if N == 0:
        print()
        return
    
    left = [-1] * N
    stack = []
    for j in range(N):
        while stack and H[stack[-1]] <= H[j]:
            stack.pop()
        if stack:
            left[j] = stack[-1]
        stack.append(j)
    
    res = [0] * (N + 2)  # res[0..N+1]
    for j in range(N):
        a = left[j] + 1
        b = j - 1
        if a <= b:
            res[a] += 1
            if b + 1 <= N:
                res[b + 1] -= 1
    
    current = 0
    c = []
    for i in range(N):
        current += res[i]
        c.append(str(current))
    
    print(' '.join(c))

if __name__ == '__main__':
    main()