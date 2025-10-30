def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    H = list(map(int, input[1:N+1]))
    
    if N == 0:
        return
    
    prefix = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix[i] = prefix[i-1] + H[i-1]
    
    t = [0] * (N + 1)
    if N >= 1:
        t[1] = H[0] + 1
    
    for i in range(2, N + 1):
        t[i] = prefix[i] + t[i-1]
    
    print(' '.join(map(str, t[1:N+1])))

if __name__ == '__main__':
    main()