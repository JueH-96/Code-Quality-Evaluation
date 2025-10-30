def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    idx = 1
    
    answers = []
    for _ in range(t):
        N = int(input_data[idx]); idx += 1
        A = list(map(int, input_data[idx:idx+N]))
        idx += N
        
        # Sum of all elements
        S = sum(A)
        # The key observation:
        # We can freely increase or decrease A_k for k >= 2 by transferring
        # from/to other indices, but we CANNOT decrease A_1 (the first element),
        # because there is no i < 1 to push A_1's "mass" into.
        #
        # Therefore, in any final non-decreasing arrangement, the first element
        # must be at least A_1.  If the final array of length N starts with
        # A_1' >= A[0] and is non-decreasing, the minimal total sum is N*A_1'.
        #
        # Hence a necessary and sufficient condition is:
        #    N*A[0] <= sum(A).
        #
        # If that holds, print "Yes", otherwise "No".
        
        if A[0] * N <= S:
            answers.append("Yes")
        else:
            answers.append("No")
    
    print("
".join(answers))

# Don't forget to call main()!
if __name__ == "__main__":
    main()