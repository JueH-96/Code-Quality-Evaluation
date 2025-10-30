import sys

def main() -> None:
    # Read two integers L and R
    try:
        L, R = map(int, sys.stdin.read().strip().split())
    except ValueError:
        # In case of malformed input, though per problem statement this shouldn't happen
        print("Invalid")
        return

    if L == 1 and R == 0:
        print("Yes")      # Only left hand raised → wants to eat takoyaki
    elif L == 0 and R == 1:
        print("No")       # Only right hand raised → doesn't want takoyaki
    else:
        print("Invalid")  # Both or neither hands raised

if __name__ == "__main__":
    main()