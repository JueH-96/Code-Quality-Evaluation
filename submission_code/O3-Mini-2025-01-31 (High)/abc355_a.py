def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    A, B = map(int, input_data)
    
    # If both witnesses designate the same person as not the culprit,
    # there remain two suspects, so we cannot determine uniquely.
    if A == B:
        print(-1)
    else:
        # There are 3 persons: sum is 1+2+3 = 6
        # Removing A and B leaves the culprit.
        culprit = 6 - (A + B)
        print(culprit)

if __name__ == '__main__':
    main()