import math

def find_A(B):
    # Check values of A from 1 to a reasonable upper bound
    A = 1
    upper_bound = math.ceil(math.log2(B))
    
    while A <= upper_bound:
        power = A ** A
        if power == B:
            return A
        if power > B:
            break  # If A^A > B, then for any larger A, A^A will be even larger
        A += 1
    
    return -1  # No solution found

B = int(input())
print(find_A(B))