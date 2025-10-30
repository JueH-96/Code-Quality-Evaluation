def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    H = list(map(int, input[1:n+1]))
    
    stack = []
    next_greater = [0] * n
    
    for i in range(n-1, -1, -1):
        while stack and H[i] >= H[stack[-1]]:
            stack.pop()
        if not stack:
            next_greater[i] = n
        else:
            next_greater[i] = stack[-1]
        stack.append(i)
    
    count = [0] * n
    for i in range(n-1, -1, -1):
        if next_greater[i] == n:
            count[i] = 1
        else:
            count[i] = 1 + count[next_greater[i]]
    
    ans = []
    for k in range(n):
        if k == n-1:
            ans.append(0)
        else:
            ans.append(count[k+1])
    
    print(' '.join(map(str, ans)))

if __name__ == '__main__':
    main()