import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    H = list(map(int, input[1:N+1]))
    
    res = [0] * N
    stack = []
    prev = 0
    
    for i in range(N):
        while stack and H[stack[-1][0]] <= H[i]:
            stack.pop()
        
        if stack:
            idx = stack[-1][1]
            res[i] = max(res[i-1] + 1, idx + H[i] + 1)
        else:
            res[i] = max(res[i-1] + 1, H[i] + 1)
        
        # Push current info to stack
        # (current index in H, res[i] value)
        stack.append((i, res[i]))
    
    print(' '.join(map(str, res)))

if __name__ == "__main__":
    main()