# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))

    # Sort the strengths in descending order
    A.sort(reverse=True)

    max_experience = 0
    defeated_count = 0

    for strength in A:
        defeated_count += 1
        max_experience += strength
        if defeated_count % 2 == 0:
            max_experience += strength

    print(max_experience)

if __name__ == "__main__":
    main()