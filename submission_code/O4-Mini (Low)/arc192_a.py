def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int,data[1:] ))
    # If already no zeros, of course "Yes".
    # The only impossible case is N=3 and all three are zero.
    if all(x==1 for x in A):
        print("Yes")
    elif N==3 and sum(A)==0:
        print("No")
    else:
        print("Yes")

if __name__ == "__main__":
    main()