# YOUR CODE HERE
def find_first_occurrence(s):
    seen = set()
    for i, char in enumerate(s, 1):
        seen.add(char)
        if len(seen) == 3:
            return i
    return len(s)

n = int(input())
s = input().strip()

print(find_first_occurrence(s))