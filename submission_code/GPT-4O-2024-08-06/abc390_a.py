# YOUR CODE HERE
def can_sort_by_one_swap(A):
    # Create a sorted version of A
    sorted_A = sorted(A)
    
    # Count the number of positions where A and sorted_A differ
    diff_count = 0
    diff_indices = []
    
    for i in range(5):
        if A[i] != sorted_A[i]:
            diff_count += 1
            diff_indices.append(i)
    
    # If there are exactly two differences, check if swapping them sorts the array
    if diff_count == 2:
        i, j = diff_indices
        # Check if swapping A[i] and A[j] sorts the array
        A[i], A[j] = A[j], A[i]
        if A == sorted_A:
            return "Yes"
    
    return "No"

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    A = list(map(int, input().strip().split()))
    print(can_sort_by_one_swap(A))