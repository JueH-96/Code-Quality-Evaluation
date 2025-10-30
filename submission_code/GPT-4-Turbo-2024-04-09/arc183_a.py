from itertools import permutations
from math import factorial, floor
from collections import Counter
import sys
input = sys.stdin.read

def main():
    data = input().strip()
    N, K = map(int, data.split())
    
    # Create the full sequence list
    full_sequence = [i for i in range(1, N + 1) for _ in range(K)]
    
    # Calculate the total number of good sequences
    # S = (N*K)! / (K!)^N
    total_permutations = factorial(N * K)
    for _ in range(N):
        total_permutations //= factorial(K)
    
    # Find the median index (1-based)
    median_index = (total_permutations + 1) // 2
    
    # Generate permutations and find the median_index-th one in lexicographical order
    # We use a counter to avoid generating all permutations if not necessary
    counter = Counter(full_sequence)
    
    # Helper function to calculate the number of permutations of a sequence given a counter
    def count_permutations(counter):
        total = factorial(sum(counter.values()))
        for count in counter.values():
            total //= factorial(count)
        return total
    
    # Generate the median permutation lexicographically
    result = []
    while counter:
        for i in sorted(counter.keys()):
            # Try to place i first and see if we can still reach the median_index
            counter[i] -= 1
            if counter[i] == 0:
                del counter[i]
            
            perms_with_i_first = count_permutations(counter)
            
            if median_index <= perms_with_i_first:
                result.append(i)
                break
            else:
                # Restore counter and update median_index
                median_index -= perms_with_i_first
                if i in counter:
                    counter[i] += 1
                else:
                    counter[i] = 1
    
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()