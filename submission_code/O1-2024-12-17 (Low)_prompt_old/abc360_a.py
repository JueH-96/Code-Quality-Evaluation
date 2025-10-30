def solve():
    S = input().strip()
    # Find the index of 'R' (rice) and 'M' (miso soup).
    # If the index of rice is less than the index of miso soup, print "Yes". Otherwise, print "No".
    if S.index('R') < S.index('M'):
        print("Yes")
    else:
        print("No")

def main():
    solve()

if __name__ == "__main__":
    main()