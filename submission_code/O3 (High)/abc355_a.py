def main():
    import sys
    
    # Read two integers A and B
    data = sys.stdin.read().strip().split()
    if not data:
        return
    A, B = map(int, data)

    # Set of all suspects
    suspects = {1, 2, 3}

    # Remove the persons who are known to be innocent
    suspects.discard(A)
    suspects.discard(B)

    # If exactly one suspect remains, print that number
    if len(suspects) == 1:
        print(suspects.pop())
    else:
        # Can't uniquely identify the culprit
        print(-1)

if __name__ == "__main__":
    main()