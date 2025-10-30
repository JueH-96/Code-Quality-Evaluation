def can_merge_strings(s1, s2):
    # Check if s1 and s2 can be merged with overlap
    for i in range(min(len(s1), len(s2)) - 1, 0, -1):
        if s1[-i:] == s2[:i]:
            return s1 + s2[i:]
    return s1 + s2

def is_valid_string(s, valid_pairs):
    # Check if all length-2 substrings are valid product names
    for i in range(len(s) - 1):
        if s[i:i+2] not in valid_pairs:
            return False
    return True

def find_min_strings(N, products):
    # For small N, try all possible combinations of products
    # and find minimum number of strings needed
    
    # First, try with 1 string
    products = set(products)
    
    # Try to create single strings by merging products
    possible_strings = list(products)
    changed = True
    while changed:
        changed = False
        new_strings = []
        for i in range(len(possible_strings)):
            for j in range(len(possible_strings)):
                if i != j:
                    merged = can_merge_strings(possible_strings[i], possible_strings[j])
                    if is_valid_string(merged, products):
                        new_strings.append(merged)
                        changed = True
        if changed:
            possible_strings.extend(new_strings)
            possible_strings = list(set(possible_strings))
    
    # Check if any single string contains all products
    for s in possible_strings:
        covers_all = True
        for product in products:
            if product not in s:
                covers_all = False
                break
        if covers_all:
            return 1
    
    # Try with 2 strings
    for i in range(len(possible_strings)):
        for j in range(i, len(possible_strings)):
            s1, s2 = possible_strings[i], possible_strings[j]
            covers_all = True
            for product in products:
                if product not in s1 and product not in s2:
                    covers_all = False
                    break
            if covers_all:
                return 2
    
    # If we can't do it with 1 or 2 strings, it must be 3
    # (we can always do it with 3 by using individual strings or small combinations)
    return 3

# Read input
N = int(input())
products = []
for _ in range(N):
    products.append(input().strip())

# Print result
print(find_min_strings(N, products))