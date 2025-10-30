def count_abc(s):
    count = 0
    for i in range(len(s) - 2):
        if s[i:i+3] == "ABC":
            count += 1
    return count

n, q = map(int, input().split())
s = list(input())
for _ in range(q):
    x, c = input().split()
    s[int(x) - 1] = c
    print(count_abc("".join(s)))