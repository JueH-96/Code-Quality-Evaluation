def count_valid_x(A, B):
    potential_x = set()
    
    # Case 1: x is the middle term
    if (A + B) % 2 == 0:
        potential_x.add((A + B) // 2)
    
    # Case 2: A is the middle term
    potential_x.add(2 * A - B)
    
    # Case 3: B is the middle term
    potential_x.add(2 * B - A)
    
    return len(potential_x)

A, B = map(int, input().split())
print(count_valid_x(A, B))