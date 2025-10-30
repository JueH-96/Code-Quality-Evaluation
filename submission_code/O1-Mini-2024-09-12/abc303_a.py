# YOUR CODE HERE
def main():
    import sys

    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    def is_similar(x, y):
        if x == y:
            return True
        if (x == '1' and y == 'l') or (x == 'l' and y == '1'):
            return True
        if (x == '0' and y == 'o') or (x == 'o' and y == '0'):
            return True
        return False

    for i in range(N):
        if not is_similar(S[i], T[i]):
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()