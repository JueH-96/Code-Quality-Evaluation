# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1:N+1]
    
    M = max(len(s) for s in S)
    
    T = [''] * M
    
    for i in range(M):
        for j in range(N):
            idx = N - j - 1
            if i < len(S[idx]):
                T[i] += S[idx][i]
            else:
                T[i] += '*'
    
    for i in range(M):
        T[i] = T[i].rstrip('*')
    
    for t in T:
        print(t)

if __name__ == "__main__":
    main()