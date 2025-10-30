def solve():
    s = input()
    n = len(s)
    substrings = set()
    for i in range(n):
        for j in range(i, n):
            substrings.add(s[i:j+1])
    print(len(substrings))

if __name__ == "__main__":
    solve()