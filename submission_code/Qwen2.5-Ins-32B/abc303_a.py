# YOUR CODE HERE
def are_similar_strings(N, S, T):
    similar_pairs = {'1': 'l', 'l': '1', '0': 'o', 'o': '0'}
    for i in range(N):
        if S[i] != T[i] and S[i] not in similar_pairs.get(T[i], ''):
            return "No"
    return "Yes"

N = int(input())
S = input().strip()
T = input().strip()

print(are_similar_strings(N, S, T))