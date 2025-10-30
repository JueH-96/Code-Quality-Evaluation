def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    D = int(data[2])
    
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+N+M]))
    
    A.sort()
    B.sort()
    
    # Two pointers approach
    max_sum = -1
    j = 0
    
    for a in A:
        # Move B's pointer to the minimum value that is within D of a
        while j < M and B[j] < a - D:
            j += 1
        # Check the current and next values if they are within the range
        if j < M and B[j] <= a + D:
            max_sum = max(max_sum, a + B[j])
        if j > 0 and B[j-1] >= a - D:
            max_sum = max(max_sum, a + B[j-1])
    
    print(max_sum)

if __name__ == '__main__':
    main()