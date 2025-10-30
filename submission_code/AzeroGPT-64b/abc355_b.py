def check_consecutive_elements(n, m, a, b):
    """
    Determines if the sorted union of two distinct integer sequences
    contains consecutive elements from the first sequence.
    
    Parameters:
    n (int): Length of the first sequence A.
    m (int): Length of the second sequence B.
    a (list): The first sequence of integers.
    b (list): The second sequence of integers.
    
    Returns:
    str: 'Yes' if there are consecutive elements from A in the sorted union, otherwise 'No'.
    """
    a_set = set(a)
    a_sorted = sorted(a)
    c_sorted = sorted(a + b)
    
    for i in range(len(a_sorted) - 1):
        if a_sorted[i] + 1 == a_sorted[i + 1]:
            if a_sorted[i] in c_sorted and a_sorted[i + 1] in c_sorted:
                ai = c_sorted.index(a_sorted[i])
                if ai + 1 < len(c_sorted) and c_sorted[ai + 1] == a_sorted[i + 1]:
                    return "Yes"
    return "No"

# Reading input
n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Solving and printing the result
print(check_consecutive_elements(n, m, a, b))