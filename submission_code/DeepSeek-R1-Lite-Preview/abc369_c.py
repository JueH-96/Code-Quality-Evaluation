def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    if N == 0:
        print(0)
        return
    A = list(map(int, data[1:N+1]))
    
    if N == 1:
        print(1)
        return
    
    D = [A[i+1] - A[i] for i in range(N-1)]
    
    total = N  # Single elements
    prev_diff = D[0]
    count = 1
    
    for i in range(1, N-1):
        if D[i] == prev_diff:
            count += 1
        else:
            total += count * (count + 1) // 2
            prev_diff = D[i]
            count = 1
    # Add the last run
    total += count * (count + 1) // 2
    print(total)

if __name__ == "__main__":
    main()