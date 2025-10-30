N, Q = map(int, input().split())
S = list(input())

def count_abc(s):
    count = 0
    for i in range(len(s)-2):
        if s[i:i+3] == ['A','B','C']:
            count += 1
    return count

for _ in range(Q):
    X, C = input().split()
    X = int(X)
    S[X-1] = C
    print(count_abc(S))