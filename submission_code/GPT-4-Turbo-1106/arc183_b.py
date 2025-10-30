from collections import defaultdict

def is_possible_to_make_identical(N, K, A, B):
    # Create a mapping from elements in B to their indices
    indices = defaultdict(list)
    for idx, value in enumerate(B):
        indices[value].append(idx)
    
    # Check if we can make A identical to B
    for idx, value in enumerate(A):
        if value not in indices:
            return False
        # Find a B index that is within K distance from the current A index
        possible = False
        for b_idx in indices[value]:
            if abs(idx - b_idx) <= K:
                possible = True
                break
        if not possible:
            return False
        # Remove the used index to prevent reuse
        indices[value].remove(b_idx)
    return True

# Read number of test cases
T = int(input().strip())

# Process each test case
for _ in range(T):
    # Read N and K
    N, K = map(int, input().strip().split())
    # Read sequences A and B
    A = list(map(int, input().strip().split()))
    B = list(map(int, input().strip().split()))
    # Check if it's possible to make A identical to B
    result = is_possible_to_make_identical(N, K, A, B)
    print("Yes" if result else "No")