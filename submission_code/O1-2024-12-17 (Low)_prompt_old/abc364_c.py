def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, X, Y = map(int, data[:3])
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+2*N]))

    sumA = sum(A)
    sumB = sum(B)

    # If we cannot exceed either X or Y, Takahashi will end up eating all N dishes
    if sumA <= X and sumB <= Y:
        print(N)
        return

    # Function to find minimal k for which the sum of the k largest values > limit
    def find_min_k_to_exceed(values, limit):
        # If the total of all values cannot exceed the limit, return infinity
        if sum(values) <= limit:
            return float('inf')
        values_sorted = sorted(values, reverse=True)
        s = 0
        for i, v in enumerate(values_sorted, start=1):
            s += v
            if s > limit:
                return i
        return float('inf')

    kA = find_min_k_to_exceed(A, X)
    kB = find_min_k_to_exceed(B, Y)

    # The answer is the smaller of the two (infinite if not possible in that dimension)
    answer = min(kA, kB)
    # If both are inf, then it implies we somehow didn't exceed (but we handled that earlier)
    print(answer if answer != float('inf') else N)

# call solve() to execute
def main():
    solve()

if __name__ == "__main__":
    main()