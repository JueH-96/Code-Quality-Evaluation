def min_operations_to_transform(N, S, T):
    if S == T:
        return 0
    
    # Create a mapping from characters in S to characters in T
    mapping = {}
    for s_char, t_char in zip(S, T):
        if s_char in mapping:
            if mapping[s_char] != t_char:
                return -1  # Conflict in mapping
        else:
            mapping[s_char] = t_char
    
    # Create a reverse mapping to check for cycles
    reverse_mapping = {}
    for s_char, t_char in mapping.items():
        if t_char in reverse_mapping:
            if reverse_mapping[t_char] != s_char:
                return -1  # Conflict in reverse mapping
        else:
            reverse_mapping[t_char] = s_char
    
    # The number of unique transformations needed is the number of unique characters in S
    # that map to different characters in T
    unique_transforms = set(mapping.values())
    
    return len(unique_transforms)

import sys

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    S = data[1]
    T = data[2]
    
    result = min_operations_to_transform(N, S, T)
    print(result)

if __name__ == "__main__":
    main()