def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = int(input_data[1])
    B = int(input_data[2])
    D = list(map(int, input_data[3:]))

    # A week has M = A + B days (0-based indexing for convenience).
    M = A + B
    
    # Compute each plan day modulo M and sort.
    R = [di % M for di in D]
    R.sort()
    
    # The problem reduces to checking whether all R[i] can fit
    # into an arc of length A (0 through A-1) for some shift.
    # Equivalently, we check if there is a contiguous sub-interval
    # of length A on the circle of circumference M that contains all R.
    #
    # A classic approach is to consider R "doubled" (concatenate
    # R plus each element of R plus M) and then use a two-pointer
    # to see if we can cover all N points in a window of length A.
    
    # Double the array with offset M.
    R2 = R + [r + M for r in R]
    
    # Two-pointer approach to find if there's any sub-interval of length A
    # containing all N points.
    j = 0
    for i in range(N):
        start_val = R2[i]
        # Move j while it is within the segment of length A
        while j < 2*N and R2[j] < start_val + A:
            j += 1
        # [i, j-1] is the largest window that starts at i with length < A in R2
        # We have (j - i) elements in this window.
        if j - i >= N:
            print("Yes")
            return
    
    print("No")

# Do not forget to call main().
main()