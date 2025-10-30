def solve():
    n = int(input())
    s = list(input())
    q = int(input())

    for _ in range(q):
        line = input().split()
        t = int(line[0])
        if t == 1:
            x = int(line[1])
            c = line[2]
            s[x-1] = c
        elif t == 2:
            for i in range(n):
                if 'A' <= s[i] <= 'Z':
                    s[i] = s[i].lower()
        elif t == 3:
            for i in range(n):
                if 'a' <= s[i] <= 'z':
                    s[i] = s[i].upper()

    print("".join(s))

if __name__ == "__main__":
    solve()