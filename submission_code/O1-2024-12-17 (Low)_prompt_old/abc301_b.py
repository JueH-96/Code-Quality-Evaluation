def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Continuously attempt to insert numbers until no adjacent pair
    # has an absolute difference other than 1.
    while True:
        inserted_anywhere = False
        for i in range(len(A) - 1):
            if abs(A[i] - A[i+1]) != 1:
                inserted_anywhere = True
                if A[i] < A[i+1]:
                    # Insert ascending series between A[i] and A[i+1]
                    to_insert = list(range(A[i]+1, A[i+1]))
                else:
                    # Insert descending series between A[i] and A[i+1]
                    to_insert = list(range(A[i]-1, A[i+1], -1))

                # Rebuild sequence with the inserted values
                A = A[:i+1] + to_insert + A[i+1:]
                break  # Restart from the beginning to check again

        # If no insertion was needed in this pass, we are done
        if not inserted_anywhere:
            break

    # Print the final sequence
    print(" ".join(map(str, A)))

# Call solve() to execute
if __name__ == "__main__":
    solve()