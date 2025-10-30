N = int(input())
S = input()

def count_repetitive_substrings(s):
    n = len(s)
    result = set()
    
    # For each starting position
    for i in range(n):
        curr_char = s[i]
        # For each length starting from position i
        for j in range(i, n):
            if s[j] == curr_char:
                result.add(curr_char * (j - i + 1))
            else:
                break
                
    return len(result)

print(count_repetitive_substrings(S))