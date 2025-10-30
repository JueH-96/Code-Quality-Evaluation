import sys
import itertools

def can_transform(s1, s2):
    """ Check if s1 can be transformed to s2 by changing exactly one character """
    diff_count = sum(1 for a, b in zip(s1, s2) if a != b)
    return diff_count == 1

def is_valid_sequence(strings):
    """ Check if the sequence of strings is valid according to the problem statement """
    return all(can_transform(strings[i], strings[i+1]) for i in range(len(strings) - 1))

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    strings = data[2:]
    
    # Generate all permutations of the strings
    for perm in itertools.permutations(strings):
        if is_valid_sequence(perm):
            print("Yes")
            return
    
    print("No")

if __name__ == "__main__":
    main()