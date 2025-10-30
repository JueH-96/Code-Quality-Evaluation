def solve():
    s = input()
    n = len(s)
    for i in range(n):
        temp = s + s[:n-i-1][::-1]
        if temp == temp[::-1]:
            print(temp)
            return

solve()