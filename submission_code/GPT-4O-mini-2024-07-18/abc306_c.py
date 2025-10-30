def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    f = {}
    
    for i in range(3 * N):
        num = A[i]
        if num not in f:
            f[num] = []
        f[num].append(i + 1)  # Store 1-based index
    
    result = []
    
    for i in range(1, N + 1):
        result.append((f[i][1], i))  # f(i) is the second occurrence (1-based index)
    
    result.sort()  # Sort by the first element of the tuple (f(i))
    
    sorted_numbers = [num for _, num in result]
    
    print(" ".join(map(str, sorted_numbers)))

if __name__ == "__main__":
    main()