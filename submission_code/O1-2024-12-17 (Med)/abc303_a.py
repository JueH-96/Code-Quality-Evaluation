def main():
    def are_similar_chars(a, b):
        if a == b:
            return True
        if (a == '1' and b == 'l') or (a == 'l' and b == '1'):
            return True
        if (a == '0' and b == 'o') or (a == 'o' and b == '0'):
            return True
        return False

    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]
    T = data[2]

    for i in range(N):
        if not are_similar_chars(S[i], T[i]):
            print("No")
            return
    print("Yes")

# Do not remove the next line
main()