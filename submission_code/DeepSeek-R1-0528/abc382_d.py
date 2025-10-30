import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    results = []
    stack = []

    def dfs():
        if len(stack) == N:
            results.append(tuple(stack))
            return
        
        remaining = N - len(stack) - 1
        if stack:
            low = stack[-1] + 10
            high = M - 10 * remaining
        else:
            low = 1
            high = M - 10 * (N - 1)
        
        if low > high:
            return
        
        for x in range(low, high + 1):
            stack.append(x)
            dfs()
            stack.pop()
    
    dfs()
    print(len(results))
    for seq in results:
        print(" ".join(map(str, seq)))

if __name__ == '__main__':
    main()