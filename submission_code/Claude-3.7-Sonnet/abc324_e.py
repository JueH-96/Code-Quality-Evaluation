def max_subsequence_length(s, t):
    """Return the maximum length of the prefix of t that can be found as a subsequence of s."""
    pos = 0
    for c in s:
        if pos < len(t) and c == t[pos]:
            pos += 1
    return pos

def is_subsequence(s, t, start=0):
    """Check if t[start:] is a subsequence of s."""
    if start >= len(t):
        return True  # An empty subsequence is always a subsequence
    pos = start
    for c in s:
        if pos < len(t) and c == t[pos]:
            pos += 1
        if pos == len(t):
            return True
    return False

def count_valid_pairs(strings, t):
    n = len(strings)
    count = 0
    
    # Precompute for each string the maximum length of the prefix of t that can be found as a subsequence
    max_prefix_len = [max_subsequence_length(s, t) for s in strings]
    
    # Precompute for each string and each position in t whether the suffix is a subsequence
    # Using a dictionary to save memory since we only need to check certain positions
    suffix_reachable = [{} for _ in range(n)]
    
    # Check for each pair if the concatenation contains t as a subsequence
    for i in range(n):
        prefix_len = max_prefix_len[i]
        for j in range(n):
            # If S_i contains the entire T
            if prefix_len == len(t):
                count += 1
            # Else check if the suffix of T from position prefix_len is a subsequence of S_j
            else:
                if prefix_len not in suffix_reachable[j]:
                    suffix_reachable[j][prefix_len] = is_subsequence(strings[j], t, prefix_len)
                if suffix_reachable[j][prefix_len]:
                    count += 1
    
    return count

import sys

def main():
    line = sys.stdin.readline().strip().split()
    n = int(line[0])
    t = line[1]
    
    strings = []
    for _ in range(n):
        strings.append(sys.stdin.readline().strip())
    
    print(count_valid_pairs(strings, t))

if __name__ == "__main__":
    main()