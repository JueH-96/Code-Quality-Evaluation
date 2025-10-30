# YOUR CODE HERE
def count_abc(s):
    return sum(1 for i in range(len(s) - 2) if s[i:i+3] == 'ABC')

N, Q = map(int, input().split())
S = list(input().strip())

for _ in range(Q):
    X, C = input().split()
    X = int(X) - 1  # Convert to 0-based index
    S[X] = C
    print(count_abc(''.join(S)))