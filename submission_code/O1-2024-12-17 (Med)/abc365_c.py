def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:]))

    # First, check if sum(A) <= M; if so, answer is "infinite"
    sumA = sum(A)
    if sumA <= M:
        print("infinite")
        return

    # Otherwise, we do a binary search to find the maximum x
    left, right = 0, max(A)
    
    def total_subsidy(x):
        # Calculate total subsidy if the limit is x
        # Sum of min(x, A_i) for i in range(N)
        s = 0
        for val in A:
            s += val if val < x else x
            if s > M:
                # No need to continue if already exceeding M
                break
        return s
    
    answer = 0
    while left <= right:
        mid = (left + right) // 2
        if total_subsidy(mid) <= M:
            answer = mid
            left = mid + 1
        else:
            right = mid - 1
    
    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()