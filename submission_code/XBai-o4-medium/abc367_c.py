def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1
    R = list(map(int, input[idx:idx+N]))
    
    result = []
    
    def dfs(pos, current, total):
        if pos == N:
            if total % K == 0:
                result.append(list(current))
            return
        for val in range(1, R[pos] + 1):
            current.append(val)
            dfs(pos + 1, current, total + val)
            current.pop()
    
    dfs(0, [], 0)
    
    for seq in result:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()