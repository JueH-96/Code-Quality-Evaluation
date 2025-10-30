def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    total = 0
    last = {}
    current_sum = 0
    
    for i in range(N):
        x = A[i]
        prev = last.get(x, -1)
        current_sum += (i - prev)
        total += current_sum
        last[x] = i
    
    print(total)

if __name__ == "__main__":
    main()