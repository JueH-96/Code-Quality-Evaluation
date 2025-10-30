def are_similar_chars(x, y):
    return x == y or (x == '1' and y == 'l') or (x == 'l' and y == '1') or (x == '0' and y == 'o') or (x == 'o' and y == '0')

def are_similar_strings(S, T):
    return all(are_similar_chars(s, t) for s, t in zip(S, T))

N = int(input().strip())
S = input().strip()
T = input().strip()

print("Yes" if are_similar_strings(S, T) else "No")