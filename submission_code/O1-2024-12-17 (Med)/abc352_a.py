def main():
    N, X, Y, Z = map(int, input().split())
    
    # If traveling inbound (X < Y), we check if Z is between X and Y.
    # If traveling outbound (X > Y), we check if Z is between Y and X.
    if (X < Y and X < Z < Y) or (X > Y and Y < Z < X):
        print("Yes")
    else:
        print("No")
        
# Do not forget to call main()!
main()