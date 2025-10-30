def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # If there is only one element, the answer is 1 (the single subarray itself).
    if N == 1:
        print(1)
        return

    total = 1  # length[0] = 1
    curr_len = 1

    # Process the array to find the length of arithmetic subarrays ending at each position
    for i in range(1, N):
        if i == 1:
            # For the second element, we have an arithmetic subarray of length 2 (A[0], A[1])
            curr_len = 2
        else:
            # Compare the differences
            if (A[i] - A[i-1]) == (A[i-1] - A[i-2]):
                curr_len += 1
            else:
                curr_len = 2  # Restart from this pair
        total += curr_len

    print(total)

# Do not forget to call main
main()