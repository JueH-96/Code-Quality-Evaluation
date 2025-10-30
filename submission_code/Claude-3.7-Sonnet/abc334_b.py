import math

def count_trees(a, m, l, r):
    # Calculate the first tree position greater than or equal to L
    j_min = math.ceil((l - a) / m)
    
    # Calculate the last tree position less than or equal to R
    j_max = (r - a) // m
    
    # Return the number of trees between these positions (inclusive)
    return max(0, j_max - j_min + 1)

a, m, l, r = map(int, input().split())
print(count_trees(a, m, l, r))