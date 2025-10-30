# YOUR CODE HERE

N, Q = map(int, input().split())
S = list(input())

def count_substring(string, sub_string):
    count = start = 0
    while start < len(string):
        pos = string.find(sub_string, start)
        if pos != -1:
            start = pos + 1
            count += 1
        else:
            break
    return count

for _ in range(Q):
    X, C = input().split()
    X = int(X) - 1
    S[X] = C
    print(count_substring(''.join(S), 'ABC'))