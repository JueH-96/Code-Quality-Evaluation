def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]
    T = data[2]

    # Define a helper function to check if two characters are similar
    def are_similar(x, y):
        if x == y:
            return True
        if (x == 'l' and y == '1') or (x == '1' and y == 'l'):
            return True
        if (x == '0' and y == 'o') or (x == 'o' and y == '0'):
            return True
        return False

    for i in range(N):
        if not are_similar(S[i], T[i]):
            print("No")
            return

    print("Yes")

# DO NOT forget to call main()!
main()