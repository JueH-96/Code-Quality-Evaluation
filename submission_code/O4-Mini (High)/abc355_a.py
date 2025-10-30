def main():
    import sys

    data = sys.stdin.read().strip().split()
    A, B = map(int, data)
    
    # The culprit must be among {1, 2, 3} but not A and not B
    possible = {1, 2, 3} - {A, B}
    
    # If there's exactly one possible culprit, print it; otherwise -1
    if len(possible) == 1:
        print(possible.pop())
    else:
        print(-1)

if __name__ == "__main__":
    main()