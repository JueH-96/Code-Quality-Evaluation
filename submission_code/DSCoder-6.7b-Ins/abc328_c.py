def count_repeating_chars(s):
    count = 0
    for i in range(len(s) - 1):
        if s[i] == s[i+1]:
            count += 1
    return count

def solve():
    N, Q = map(int, input().split())
    S = input()
    queries = [tuple(map(int, input().split())) for _ in range(Q)]

    for l, r in queries:
        substring = S[l-1:r]
        print(count_repeating_chars(substring))

solve()