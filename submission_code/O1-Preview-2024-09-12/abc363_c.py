# YOUR CODE HERE
import sys
from collections import Counter

import threading

def main():
    import sys

    sys.setrecursionlimit(1 << 25)

    N_K_line = sys.stdin.readline().strip()
    while N_K_line == '':
        N_K_line = sys.stdin.readline().strip()
    N_str_line = sys.stdin.readline().strip()
    while N_str_line == '':
        N_str_line = sys.stdin.readline().strip()
    N_str_line = N_str_line.strip()

    N_K = N_K_line.split()
    N = int(N_K[0])
    K = int(N_K[1])
    S = N_str_line.strip()

    counts = Counter(S)

    total_count = [0]

    def dfs(curr_permutation, counts_remaining):
        if len(curr_permutation) == N:
            total_count[0] += 1
            return
        for c in counts_remaining:
            if counts_remaining[c] > 0:
                # Add character c to the permutation
                counts_remaining[c] -=1
                curr_permutation.append(c)
                # Check for palindrome of length K at the end
                if len(curr_permutation) >= K:
                    last_K = curr_permutation[-K:]
                    if last_K == last_K[::-1]:
                        # Palindrome found, prune this branch
                        curr_permutation.pop()
                        counts_remaining[c] += 1
                        continue
                # Proceed recursively
                dfs(curr_permutation, counts_remaining)
                # Backtracking
                curr_permutation.pop()
                counts_remaining[c] +=1

    curr_permutation = []

    dfs(curr_permutation, counts.copy())

    print(total_count[0])

if __name__ == '__main__':
    threading.Thread(target=main).start()