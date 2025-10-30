def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    A = list(map(int, data[idx:idx+N]))
    idx += N
    B = list(map(int, data[idx:idx+M]))
    
    A.sort()
    B.sort()
    
    i = j = total = count = 0
    while i < N and j < M:
        if A[i] >= B[j]:
            total += A[i]
            count += 1
            i += 1
            j += 1
        else:
            i += 1
    
    if count == M:
        print(total)
    else:
        print(-1)

if __name__ == "__main__":
    main()