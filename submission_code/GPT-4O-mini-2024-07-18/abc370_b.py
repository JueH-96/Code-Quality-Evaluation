def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    A = []
    
    # Read the transformation matrix
    for i in range(N):
        row = list(map(int, data[i + 1].split()))
        A.append(row)
    
    # Start with element 1 (index 0)
    current_element = 1
    
    # Combine with elements 1 to N
    for j in range(1, N + 1):
        if current_element >= j:
            current_element = A[current_element - 1][j - 1]
        else:
            current_element = A[j - 1][current_element - 1]
    
    print(current_element)

if __name__ == "__main__":
    main()