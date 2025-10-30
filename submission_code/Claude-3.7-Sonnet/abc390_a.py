def can_sort_with_one_swap(A):
    count = 0
    for i in range(4):  # Since A has 5 elements, there are 4 adjacent pairs
        A_copy = A.copy()
        A_copy[i], A_copy[i+1] = A_copy[i+1], A_copy[i]  # Swap adjacent elements
        if A_copy == [1, 2, 3, 4, 5]:  # Check if sorted
            count += 1
    
    return count == 1  # We want exactly one operation to sort A

# Read the input
A = list(map(int, input().split()))

# Check if A can be sorted with exactly one swap
if can_sort_with_one_swap(A):
    print("Yes")
else:
    print("No")