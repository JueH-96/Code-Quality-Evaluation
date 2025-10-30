# YOUR CODE HERE
import itertools

def can_rearrange_to_conforming_sequence(strings):
    N = len(strings)
    
    # Generate all permutations of the strings
    for perm in itertools.permutations(strings):
        valid = True
        # Check each consecutive pair in the permutation
        for i in range(N - 1):
            diff_count = 0
            for a, b in zip(perm[i], perm[i + 1]):
                if a != b:
                    diff_count += 1
            # If the difference is not exactly one, this permutation is invalid
            if diff_count != 1:
                valid = False
                break
        # If we found a valid permutation, return True
        if valid:
            return True
    # If no valid permutation was found, return False
    return False

def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    N = int(data[0])
    M = int(data[1])
    strings = data[2:]
    
    if can_rearrange_to_conforming_sequence(strings):
        print("Yes")
    else:
        print("No")