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
    
    # We need to find the first day X >= 1 where the total pills <= K
    # The total pills on day X is the sum of b_i for all i where a_i >= X
    
    # To find the first X, we can perform a binary search on X
    # The search range is from 1 to the maximum a_i + 1
    
    # Find the maximum a_i
    max_a = max(a)
    
    # Binary search between 1 and max_a + 1
    left = 1
    right = max_a + 1
    
    result = right  # Initialize with the maximum possible
    
    while left <= right:
        mid = (left + right) // 2
        total = 0
        for i in range(N):
            if a[i] >= mid:
                total += b[i]
        if total <= K:
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    
    print(result)

if __name__ == "__main__":
    main()