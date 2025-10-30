def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    H = list(map(int, input[1:N+1]))
    
    total = 0
    res = []
    for i in range(N):
        total += H[i]
        res.append(str(total + (i + 1)))
    
    print(' '.join(res))

if __name__ == '__main__':
    main()