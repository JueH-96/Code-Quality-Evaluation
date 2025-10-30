def main():
    import sys
    S1, S2 = sys.stdin.read().strip().split()
    # Possible bad oysters
    possible = set([1, 2, 3, 4])
    # Takahashi ate oysters 1 and 2
    if S1 == "sick":
        possible &= {1, 2}
    else:
        possible -= {1, 2}
    # Aoki ate oysters 1 and 3
    if S2 == "sick":
        possible &= {1, 3}
    else:
        possible -= {1, 3}
    # Exactly one culprit remains
    print(possible.pop())

if __name__ == "__main__":
    main()