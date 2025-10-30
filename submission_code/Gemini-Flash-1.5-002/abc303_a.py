# YOUR CODE HERE
def solve():
    n = int(input())
    s = input()
    t = input()

    def are_similar(char1, char2):
        if char1 == char2:
            return True
        if (char1 == '1' and char2 == 'l') or (char1 == 'l' and char2 == '1'):
            return True
        if (char1 == '0' and char2 == 'o') or (char1 == 'o' and char2 == '0'):
            return True
        return False

    for i in range(n):
        if not are_similar(s[i], t[i]):
            print("No")
            return
    print("Yes")

solve()