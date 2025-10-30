# YOUR CODE HERE
import sys

def determine_middle_brother(S_AB, S_AC, S_BC):
    if S_AB == '<':
        if S_AC == '<':
            if S_BC == '<':
                return 'B'
            else:
                return 'C'
        else:
            if S_BC == '<':
                return 'A'
            else:
                return 'B'
    else:
        if S_AC == '<':
            if S_BC == '<':
                return 'A'
            else:
                return 'C'
        else:
            if S_BC == '<':
                return 'B'
            else:
                return 'A'

def main():
    input = sys.stdin.read().strip()
    S_AB, S_AC, S_BC = input.split()
    print(determine_middle_brother(S_AB, S_AC, S_BC))

if __name__ == "__main__":
    main()