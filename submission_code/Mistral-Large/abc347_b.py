import sys

def count_unique_substrings(s):
    n = len(s)
    substrings = set()

    for i in range(n):
        for j in range(i + 1, n + 1):
            substrings.add(s[i:j])

    return len(substrings)

if __name__ == "__main__":
    input = sys.stdin.read()
    S = input.strip()
    result = count_unique_substrings(S)
    print(result)