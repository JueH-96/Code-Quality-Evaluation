import sys

def solve():
    # Read N and M from the first line
    N, M = map(int, sys.stdin.readline().split())
    
    # Read sequence A from the second line
    A = list(map(int, sys.stdin.readline().split()))
    
    # Read sequence B from the third line
    B = list(map(int, sys.stdin.readline().split()))

    # --- Calculate first_match_indices (leftmost greedy subsequence) ---
    # This list will store the indices in A that form the first (leftmost) match for B.
    first_match_indices = []
    a_ptr = 0 # Pointer for sequence A
    b_ptr = 0 # Pointer for sequence B

    # Iterate through A to find elements matching B, greedily taking the first available.
    while a_ptr < N and b_ptr < M:
        if A[a_ptr] == B[b_ptr]:
            first_match_indices.append(a_ptr)
            b_ptr += 1 # Move to the next element in B
        a_ptr += 1 # Always move to the next element in A

    # If b_ptr is less than M, it means we couldn't find all elements of B in A.
    # So B is not a subsequence of A, meaning 0 matches.
    if b_ptr < M:
        print("No")
        return

    # --- Calculate last_match_indices (rightmost greedy subsequence) ---
    # This list will store the indices in A that form the last (rightmost) match for B.
    # Initialize with M placeholders; the indices will be filled from right to left.
    last_match_indices = [0] * M 
    a_ptr = N - 1 # Pointer for sequence A, starting from the end
    b_ptr = M - 1 # Pointer for sequence B, starting from the end

    # Iterate through A from right to left to find elements matching B, greedily taking the last available.
    while a_ptr >= 0 and b_ptr >= 0:
        if A[a_ptr] == B[b_ptr]:
            last_match_indices[b_ptr] = a_ptr
            b_ptr -= 1 # Move to the previous element in B
        a_ptr -= 1 # Always move to the previous element in A
    
    # This check is technically not needed if the first `b_ptr < M` check passed,
    # because if B is a subsequence, both greedy methods will find it.
    # However, it doesn't hurt and makes the logic symmetric.
    if b_ptr >= 0:
        print("No")
        return

    # --- Compare the two sets of indices ---
    # If the leftmost greedy subsequence and the rightmost greedy subsequence
    # use different sets of indices, it implies that there must be at least two ways
    # to form B as a subsequence of A.
    if first_match_indices != last_match_indices:
        print("Yes")
    else:
        # If they are identical, it means there is only one unique way to form B.
        print("No")

# Call the solve function to run the program
solve()