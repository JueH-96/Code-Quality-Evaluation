def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    
    A = [int(c) for c in S]
    
    total = 0
    
    for i in range(N):
        current = A[i]
        total += current
        for j in range(i+1, N):
            current = 1 if (current == 0 or A[j] == 0) else 0
            total += current
    
    print(total)

if __name__ == "__main__":
    main()