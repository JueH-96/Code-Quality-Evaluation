def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    current = 0
    min_sum = 0
    total = 0
    
    for a in A:
        current += a
        if current < min_sum:
            min_sum = current
    
    S = max(0, -min_sum)
    final = S + sum(A)
    print(final)

if __name__ == "__main__":
    main()