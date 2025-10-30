def solve():
    n = int(input())
    s = input()
    q = int(input())

    current_s = list(s)

    for _ in range(q):
        line = input().split()
        t = int(line[0])
        if t == 1:
            x = int(line[1])
            c = line[2]
            current_s[x-1] = c
        elif t == 2:
            for i in range(len(current_s)):
                if 'A' <= current_s[i] <= 'Z':
                    current_s[i] = current_s[i].lower()
        elif t == 3:
            for i in range(len(current_s)):
                if 'a' <= current_s[i] <= 'z':
                    current_s[i] = current_s[i].upper()

    print("".join(current_s))

if __name__ == '__main__':
    solve()