def main():
    import sys
    
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:]))

    A.sort()
    
    j = 0
    max_count = 0
    for i in range(N):
        while j < N and A[j] < A[i] + M:
            j += 1
        max_count = max(max_count, j - i)
    
    print(max_count)

if __name__ == "__main__":
    main()