# YOUR CODE HERE
def are_similar(s, t):
    similar_pairs = {'1': 'l', 'l': '1', '0': 'o', 'o': '0'}
    for i in range(len(s)):
        if s[i] != t[i] and (s[i] not in similar_pairs or similar_pairs[s[i]] != t[i]):
            return "No"
    return "Yes"

N = int(input())
S = input()
T = input()

print(are_similar(S, T))