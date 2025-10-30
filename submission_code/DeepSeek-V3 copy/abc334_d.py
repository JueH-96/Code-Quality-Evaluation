def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    R = list(map(int, data[2:2+N]))
    queries = list(map(int, data[2+N:2+N+Q]))
    
    R.sort()
    
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + R[i]
    
    for X in queries:
        low = 0
        high = N
        answer = 0
        while low <= high:
            mid = (low + high) // 2
            if prefix_sum[mid] <= X:
                answer = mid
                low = mid + 1
            else:
                high = mid - 1
        print(answer)

if __name__ == "__main__":
    main()