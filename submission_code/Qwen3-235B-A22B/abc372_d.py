def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    h = list(map(int, data[1:n+1]))
    
    L = [-1] * n
    stack = []
    for j in range(n):
        while stack and h[stack[-1]] < h[j]:
            stack.pop()
        if stack:
            L[j] = stack[-1]
        else:
            L[j] = -1
        stack.append(j)
    
    diff = [0] * (n + 2)  # To handle end+1 up to n
    for j in range(n):
        start = max(L[j], 0)
        end = j - 1
        if start > end:
            continue
        diff[start] += 1
        if end + 1 < n:
            diff[end + 1] -= 1
    
    res = [0] * n
    current = 0
    for i in range(n):
        current += diff[i]
        res[i] = current
    
    print(' '.join(map(str, res)))

if __name__ == '__main__':
    main()