from collections import defaultdict

def soln():
    N, T = input().split()
    N = int(N)
    
    # Read the strings S_i
    S = [input() for _ in range(N)]
    
    # Create a dictionary to store the indices of each character in each string
    char_indices = defaultdict(lambda: [])
    for i, s in enumerate(S):
        for j, c in enumerate(s):
            char_indices[c].append((i, j))
    
    # Count the number of pairs (i, j) that satisfy the condition
    count = 0
    for i in range(N):
        for j in range(N):
            # Check if the concatenation of S_i and S_j contains T as a subsequence
            i_idx, j_idx = 0, 0
            for c in T:
                if c not in char_indices:
                    break
                found = False
                for (i_, j_) in char_indices[c]:
                    if i_ == i and j_ > i_idx or i_ == j and j_ > j_idx:
                        i_idx = j_ + 1
                        j_idx = j_ + 1
                        found = True
                        break
                if not found:
                    break
            else:
                count += 1
    
    print(count)

soln()