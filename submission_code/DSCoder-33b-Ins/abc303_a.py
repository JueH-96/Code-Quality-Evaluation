# YOUR CODE HERE

def are_similar(s, t):
    similar_chars = {'1': 'l', 'l': '1', '0': 'o', 'o': '0'}
    for i in range(len(s)):
        if s[i] != t[i] and (s[i] not in similar_chars or similar_chars[s[i]] != t[i]):
            return False
    return True

n = int(input())
s = input()
t = input()

if are_similar(s, t):
    print('Yes')
else:
    print('No')