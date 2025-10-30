def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    H = list(map(int, data[1:N+1]))
    
    left = [-1] * N
    stack = []
    for j in range(N):
        while stack and H[stack[-1]] <= H[j]:
            stack.pop()
        if stack:
            left[j] = stack[-1]
        else:
            left[j] = -1
        stack.append(j)
    
    diff = [0] * N
    for j in range(N):
        L = max(left[j], 0)
        R = j - 1
        if L <= R:
            diff[L] += 1
            if R + 1 < N:
                diff[R + 1] -= 1
    
    c = []
    current = 0
    for i in range(N):
        current += diff[i]
        c.append(current)
    
    print(' '.join(map(str, c)))

if __name__ == '__main__':
    main()