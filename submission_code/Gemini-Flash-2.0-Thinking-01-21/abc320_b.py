def solve():
    s = input()
    n = len(s)
    max_len = 0
    for i in range(n):
        for j in range(i, n):
            substring = s[i:j+1]
            if substring == substring[::-1]:
                max_len = max(max_len, len(substring))
    print(max_len)

if __name__ == "__main__":
    solve()