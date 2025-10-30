def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+M]))
    
    # Precompute the answers for each day
    result = [0] * N
    current = 0
    
    for i in range(N):
        day = i + 1
        while current < M and A[current] < day:
            current += 1
        if current < M:
            result[i] = max(0, A[current] - day)
        else:
            result[i] = 0
    
    for res in result:
        print(res)

if __name__ == "__main__":
    main()