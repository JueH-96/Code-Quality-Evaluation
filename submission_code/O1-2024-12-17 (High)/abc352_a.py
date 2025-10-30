def main():
    N, X, Y, Z = map(int, input().split())
    # The train route from X to Y (whether inbound or outbound) will include
    # every station from X to Y in increasing or decreasing order.
    # We just need to check if Z lies strictly between X and Y
    # (i.e., min(X, Y) < Z < max(X, Y)).
    if min(X, Y) < Z < max(X, Y):
        print("Yes")
    else:
        print("No")

# Do not forget to call main()
if __name__ == "__main__":
    main()