def solve():
    n, q = map(int, input().split())
    s = list(input())
    
    def count_abc(s):
        count = 0
        for i in range(len(s) - 2):
            if s[i:i+3] == ['A', 'B', 'C']:
                count += 1
        return count

    for _ in range(q):
        x, c = input().split()
        x = int(x)
        s[x-1] = c
        print(count_abc(s))

solve()