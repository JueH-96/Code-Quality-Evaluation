import sys

def count_unique_repeated_substrings(s):
    unique_substrings = set()
    n = len(s)
    i = 0
    
    while i < n:
        start = i
        while i < n - 1 and s[i] == s[i + 1]:
            i += 1
        end = i + 1
        repeated_char = s[start]
        for length in range(1, end - start + 1):
            unique_substrings.add(repeated_char * length)
        i += 1
    
    return len(unique_substrings)

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    result = count_unique_repeated_substrings(S)
    print(result)