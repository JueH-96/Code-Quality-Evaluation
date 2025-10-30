def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    
    a = []
    b = []
    index = 2
    for _ in range(N):
        a.append(int(data[index]))
        b.append(int(data[index+1]))
        index += 2
    
    # Binary search for the first day where total pills <= K
    left = 1
    right = max(a) + 1  # Since a_i can be up to 1e9, but we need to cover all possible days
    
    result = right  # Initialize with a large value
    
    while left <= right:
        mid = (left + right) // 2
        total = 0
        for i in range(N):
            if mid <= a[i]:
                total += b[i]
        if total <= K:
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    
    print(result)

if __name__ == "__main__":
    main()