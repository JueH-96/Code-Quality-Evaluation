from collections import defaultdict

def count_pairs(N, A):
    # Create a dictionary to store the indices of each number
    indices = defaultdict(list)
    for i, num in enumerate(A):
        indices[num].append(i)

    # Count the number of valid pairs
    count = 0
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            a_indices = indices[a]
            b_indices = indices[b]
            if len(a_indices) == 2 and len(b_indices) == 2 and \
               abs(a_indices[0] - a_indices[1]) > 1 and \
               abs(b_indices[0] - b_indices[1]) > 1:
                count += 1
    return count

T = int(input())
for _ in range(T):
    N = int(input())
    A = [int(x) for x in input().split()]
    print(count_pairs(N, A))