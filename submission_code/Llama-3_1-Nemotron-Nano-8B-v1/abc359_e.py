def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    H = list(map(int, input[1:N+1]))
    
    ans = [0] * N
    current = 0
    for i in range(N):
        next_val = H[i] + 1
        current += H[i]
        if H[i] > current - H[i]:
            current += H[i] + 1
        else:
            current += 1
        ans[i] = current
    
    print(' '.join(map(str, ans)))

if __name__ == "__main__":
    main()