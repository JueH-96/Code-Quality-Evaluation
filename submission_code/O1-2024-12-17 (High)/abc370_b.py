def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    idx = 1
    
    # Read the triangular matrix into a list of lists.
    B = []
    for i in range(1, N + 1):
        row = list(map(int, data[idx : idx + i]))
        idx += i
        B.append(row)
    
    # Start with element 1.
    current_element = 1
    
    # Combine current_element with elements 1..N in order.
    for i in range(1, N + 1):
        if current_element >= i:
            current_element = B[current_element - 1][i - 1]
        else:
            current_element = B[i - 1][current_element - 1]
    
    # Print the final element obtained.
    print(current_element)

# Do not forget to call main().
if __name__ == "__main__":
    main()