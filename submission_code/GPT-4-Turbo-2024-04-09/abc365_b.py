def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Create a list of tuples (value, index)
    indexed_A = [(A[i], i + 1) for i in range(N)]
    
    # Sort by value
    indexed_A.sort(reverse=True)
    
    # The second element in the sorted list will be the second largest
    # We need its original index
    print(indexed_A[1][1])

if __name__ == "__main__":
    main()