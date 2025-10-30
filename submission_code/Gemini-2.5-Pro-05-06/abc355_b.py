def main():
    N, M = map(int, input().split())
    A_list = list(map(int, input().split()))
    B_list = list(map(int, input().split()))

    # Store elements of A in a set for efficient lookup
    set_A = set(A_list)

    # Combine elements of A and B
    C_combined = A_list + B_list
    
    # Sort the combined list to form sequence C
    C_sorted = sorted(C_combined)

    found_consecutive_from_A = False
    
    # Iterate through C to find if any two consecutive elements are both from A
    # The loop needs to check pairs (C_sorted[i], C_sorted[i+1])
    # So, i ranges from 0 to len(C_sorted) - 2.
    # len(C_sorted) is N + M. Since N, M >= 1, N + M >= 2.
    # Thus, len(C_sorted) - 1 is at least 1, meaning the loop range is valid.
    if len(C_sorted) >= 2: # This check is true if N+M >= 2
        for i in range(len(C_sorted) - 1):
            element1 = C_sorted[i]
            element2 = C_sorted[i+1]

            if element1 in set_A and element2 in set_A:
                found_consecutive_from_A = True
                break
    
    if found_consecutive_from_A:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()