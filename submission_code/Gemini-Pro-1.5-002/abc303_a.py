# YOUR CODE HERE
def solve():
    n = int(input())
    s = input()
    t = input()

    def are_similar(c1, c2):
        if c1 == c2:
            return True
        if (c1 == '1' and c2 == 'l') or (c1 == 'l' and c2 == '1'):
            return True
        if (c1 == '0' and c2 == 'o') or (c1 == 'o' and c2 == '0'):
            return True
        return False

    for i in range(n):
        if not are_similar(s[i], t[i]):
            print("No")
            return

    print("Yes")

solve()