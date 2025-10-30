def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    A.sort()
    
    left = 0
    max_count = 0
    
    for right in range(N):
        while A[right] - A[left] >= M:
            left += 1
        current_count = right - left + 1
        if current_count > max_count:
            max_count = current_count
    
    print(max_count)

if __name__ == "__main__":
    main()