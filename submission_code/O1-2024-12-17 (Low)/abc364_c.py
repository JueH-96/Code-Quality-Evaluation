def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, X, Y = map(int, input_data[:3])
    A = list(map(int, input_data[3:3+N]))
    B = list(map(int, input_data[3+N:3+2*N]))
    
    # Helper function to find the earliest k such that the sum of the top k elements > limit
    # If we never exceed limit, return N+1 (meaning "cannot exceed").
    def earliest_exceed_top_k(arr, limit):
        arr_sorted = sorted(arr, reverse=True)
        total = 0
        for i, val in enumerate(arr_sorted, start=1):
            total += val
            if total > limit:
                return i
        return N+1  # never exceeded
    
    kA = earliest_exceed_top_k(A, X)
    kB = earliest_exceed_top_k(B, Y)
    ans = min(kA, kB)
    
    # If ans is larger than N, it means we never exceed => end up eating all N dishes
    if ans > N:
        ans = N
    
    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()