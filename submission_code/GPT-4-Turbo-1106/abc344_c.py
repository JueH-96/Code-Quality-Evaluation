from bisect import bisect_left

def can_find_triplet_sum(A, B, C, X):
    # Precompute all possible sums of elements from A and B
    AB_sums = sorted(a + b for a in A for b in B)
    
    # Check if there's a combination that sums up to X
    for x in X:
        # Binary search for the difference between x and elements of C in AB_sums
        for c in C:
            if bisect_left(AB_sums, x - c) < len(AB_sums) and AB_sums[bisect_left(AB_sums, x - c)] == x - c:
                yield 'Yes'
                break
        else:
            yield 'No'

# Read input from stdin
N = int(input().strip())
A = list(map(int, input().strip().split()))
M = int(input().strip())
B = list(map(int, input().strip().split()))
L = int(input().strip())
C = list(map(int, input().strip().split()))
Q = int(input().strip())
X = list(map(int, input().strip().split()))

# Solve the problem and write the answer to stdout
for answer in can_find_triplet_sum(A, B, C, X):
    print(answer)