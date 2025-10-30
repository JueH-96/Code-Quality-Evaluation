# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    P = int(data[2])
    L = list(map(int, data[3:3+N]))
    
    # Check if the condition is already satisfied
    count = 0
    for l in L:
        if l >= T:
            count += 1
    if count >= P:
        print(0)
        return
    
    # Binary search for the minimum number of days
    left = 0
    right = 100000  # A large enough number
    
    while left < right:
        mid = (left + right) // 2
        cnt = 0
        for l in L:
            if l + mid >= T:
                cnt += 1
        if cnt >= P:
            right = mid
        else:
            left = mid + 1
    
    print(left)

if __name__ == "__main__":
    main()