def solve():
    n = int(input())
    s = input()
    seen = set()
    for i, c in enumerate(s):
        seen.add(c)
        if len(seen) == 3:
            print(i + 1)
            break

if __name__ == "__main__":
    solve()