def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Create a list of (value, original_index)
    indexed_A = [(A[i], i) for i in range(N)]
    
    # Sort by value descending
    indexed_A.sort(key=lambda x: x[0], reverse=True)
    
    # The second largest element is in indexed_A[1]
    # original_index is stored in indexed_A[1][1]
    # We output the 1-based index, so add 1
    print(indexed_A[1][1] + 1)

def main():
    solve()

if __name__ == "__main__":
    main()