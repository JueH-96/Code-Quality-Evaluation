def main():
    N, X, Y, Z = map(int, input().split())
    # If traveling inbound (X < Y), train stops at all stations from X to Y inclusive.
    # If traveling outbound (X > Y), train stops at all stations from X down to Y inclusive.
    if X < Y:
        if X <= Z <= Y:
            print("Yes")
        else:
            print("No")
    else:  # X > Y
        if Y <= Z <= X:
            print("Yes")
        else:
            print("No")

# Call the main function
main()