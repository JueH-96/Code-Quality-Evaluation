import sys

def check_conditions(S):
    # Check if all elements are between 100 and 675, inclusive
    if not all(100 <= s <= 675 for s in S):
        return "No"

    # Check if all elements are multiples of 25
    if not all(s % 25 == 0 for s in S):
        return "No"

    # Check if the sequence is monotonically non-decreasing
    for i in range(1, len(S)):
        if S[i] < S[i - 1]:
            return "No"

    return "Yes"

def main():
    input = sys.stdin.read()
    S = list(map(int, input.split()))
    result = check_conditions(S)
    sys.stdout.write(result + "
")

if __name__ == "__main__":
    main()