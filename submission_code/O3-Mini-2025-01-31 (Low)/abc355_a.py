def main():
    import sys
    input_data = sys.stdin.read().split()
    A = int(input_data[0])
    B = int(input_data[1])
    
    # When the two witnesses exclude the same person, two suspects remain. Unique identification impossible.
    if A == B:
        print(-1)
        return
    
    # With two distinct persons excluded, the remaining one is the culprit.
    # The total of the numbers 1, 2, 3 is 6
    culprit = 6 - (A + B)
    print(culprit)

if __name__ == '__main__':
    main()