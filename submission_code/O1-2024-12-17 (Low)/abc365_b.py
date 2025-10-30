def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Pair each element with its 1-based index, e.g. (value, index)
    indexed_A = [(val, i) for i, val in enumerate(A, start=1)]
    
    # Sort by value descending
    indexed_A.sort(key=lambda x: x[0], reverse=True)
    
    # The second largest element is at index 1 after the sort
    print(indexed_A[1][1])

# Do not forget to call main()
main()