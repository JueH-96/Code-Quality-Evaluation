def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    A = list(map(int, input[1:n+1]))
    swaps = []
    for i in range(1, n+1):
        while A[i-1] != i:
            x = A[i-1]
            swaps.append((i, x))
            A[i-1], A[x-1] = A[x-1], A[i-1]
    print(len(swaps))
    for s in swaps:
        print(s[0], s[1])
        
if __name__ == "__main__":
    main()