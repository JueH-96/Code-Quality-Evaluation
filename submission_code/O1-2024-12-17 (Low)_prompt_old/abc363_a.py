def solve():
    R = int(input().strip())
    if R < 100:
        # Currently 1 caret, next threshold is 100 for 2 carets
        print(100 - R)
    elif R < 200:
        # Currently 2 carets, next threshold is 200 for 3 carets
        print(200 - R)
    else:
        # Currently 3 carets, next threshold is 300 for 4 carets
        print(300 - R)

def main():
    solve()

if __name__ == "__main__":
    main()