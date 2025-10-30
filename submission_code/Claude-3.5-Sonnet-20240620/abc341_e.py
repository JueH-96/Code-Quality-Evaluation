# YOUR CODE HERE
def is_good_string(s):
    return all(s[i] != s[i+1] for i in range(len(s)-1))

N, Q = map(int, input().split())
S = list(input().strip())

for _ in range(Q):
    query = input().split()
    query_type = int(query[0])
    L, R = map(int, query[1:])
    L -= 1  # Convert to 0-based indexing
    R -= 1

    if query_type == 1:
        for i in range(L, R+1):
            S[i] = '1' if S[i] == '0' else '0'
    else:  # query_type == 2
        substring = S[L:R+1]
        print("Yes" if is_good_string(substring) else "No")