def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    H = list(map(int, input[1:N+1]))
    
    stack = []
    prev_greater = [-1] * N
    
    for j in range(N-1, -1, -1):
        h = H[j]
        while stack and H[stack[-1]] < h:
            stack.pop()
        if stack:
            prev_greater[j] = stack[-1]
        else:
            prev_greater[j] = -1
        stack.append(j)
    
    diff = [0] * (N + 2)
    
    for j in range(N):
        pg = prev_greater[j]
        a = pg + 1 if pg != -1 else 0
        b = j - 1
        if a <= b:
            diff[a] += 1
            diff[b + 1] -= 1
    
    answer = []
    current = 0
    for i in range(N):
        current += diff[i]
        answer.append(str(current))
    
    print(' '.join(answer))

if __name__ == '__main__':
    main()