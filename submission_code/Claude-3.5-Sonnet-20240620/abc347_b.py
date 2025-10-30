# YOUR CODE HERE
def count_unique_substrings(s):
    substrings = set()
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n + 1):
            substrings.add(s[i:j])
    return len(substrings)

s = input().strip()
print(count_unique_substrings(s))