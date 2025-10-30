from collections import defaultdict

def count_subsequences(A, B):
    N, M = len(A), len(B)
    if M > N:
        return "No"

    # Create a dictionary to store the indices of each element in A
    indices = defaultdict(list)
    for i, a in enumerate(A):
        indices[a].append(i)

    # Check if there are at least two subsequences of A that match B
    count = 0
    prev_idx = -1
    for b in B:
        if b not in indices:
            return "No"
        found = False
        for idx in indices[b]:
            if idx > prev_idx:
                prev_idx = idx
                found = True
                count += 1
                break
        if not found:
            return "No"

    return "Yes" if count >= 2 else "No"

# Read input from stdin
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Solve the problem and write the output to stdout
print(count_subsequences(A, B))