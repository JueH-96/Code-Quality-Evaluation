# YOUR CODE HERE
N = int(input())
S = input()

def find_first_occurrence(S):
    seen = set()
    for i, char in enumerate(S):
        seen.add(char)
        if len(seen) == 3:
            return i + 1

print(find_first_occurrence(S))