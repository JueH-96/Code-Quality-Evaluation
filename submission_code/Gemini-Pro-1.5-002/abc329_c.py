# YOUR CODE HERE
def solve():
    n = int(input())
    s = input()
    ans = 0
    i = 0
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        length = j - i
        ans += length * (length + 1) // 2
        i = j
    print(ans)

solve()