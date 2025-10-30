def main():
    # Read integers A and B
    A, B = map(int, input().split())
    # Compute sum
    s = A + B
    # We need to print any number from 0 to 9 that is NOT equal to s.
    # A simple way is to check if s != 9; if not, print 9. Otherwise, print 0.
    if s != 9:
        print(9)
    else:
        print(0)

# Do not forget to call the main() function
if __name__ == "__main__":
    main()