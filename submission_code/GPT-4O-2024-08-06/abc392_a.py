# YOUR CODE HERE
def can_form_product(A):
    # Sort the array to make it easier to check permutations
    A.sort()
    
    # Check if any permutation satisfies B_1 * B_2 = B_3
    # Since A is sorted, we only need to check:
    # A[0] * A[1] == A[2]
    if A[0] * A[1] == A[2]:
        return "Yes"
    else:
        return "No"

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    # Read input
    A = list(map(int, input().strip().split()))
    # Output the result
    print(can_form_product(A))