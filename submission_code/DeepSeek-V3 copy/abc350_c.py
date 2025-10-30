def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    swaps = []
    for i in range(N):
        while A[i] != i + 1:
            j = A[i] - 1
            A[i], A[j] = A[j], A[i]
            swaps.append((i+1, j+1))
    
    print(len(swaps))
    for swap in swaps:
        print(swap[0], swap[1])

if __name__ == "__main__":
    main()