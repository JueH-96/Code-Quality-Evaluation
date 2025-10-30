def main():
    import sys
    
    input = sys.stdin.readline
    N = int(input().strip())
    S = input().strip()
    T = input().strip()
    
    def similar(x, y):
        # identical characters
        if x == y:
            return True
        # '1' and 'l' are similar
        if (x == '1' and y == 'l') or (x == 'l' and y == '1'):
            return True
        # '0' and 'o' are similar
        if (x == '0' and y == 'o') or (x == 'o' and y == '0'):
            return True
        return False

    for a, b in zip(S, T):
        if not similar(a, b):
            print("No")
            return

    print("Yes")


if __name__ == "__main__":
    main()