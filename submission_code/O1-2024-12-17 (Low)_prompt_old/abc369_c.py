def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # If there's only one element, the answer is 1 (the single-element subsequence).
    if N == 1:
        print(1)
        return

    total = 1      # Count for the first element alone
    curr_len = 1   # Current length of consecutive AP-ending at the current index

    for i in range(1, N):
        diff = A[i] - A[i - 1]
        if i == 1:
            # For the second element, automatically form a length-2 AP
            curr_len = 2
        else:
            # Compare current difference with the previous difference
            if diff == prev_diff:
                curr_len += 1
            else:
                # Reset to 2 because any two consecutive elements form an AP
                curr_len = 2
        total += curr_len
        prev_diff = diff

    print(total)

# Call solve() to read input, compute and print result.
def main():
    solve()

if __name__ == "__main__":
    main()