# YOUR CODE HERE
def count_consecutive_pairs(s, l, r):
    count = 0
    for i in range(l-1, r-1):
        if s[i] == s[i+1]:
            count += 1
    return count

N, Q = map(int, input().split())
S = input().strip()

for _ in range(Q):
    l, r = map(int, input().split())
    print(count_consecutive_pairs(S, l, r))