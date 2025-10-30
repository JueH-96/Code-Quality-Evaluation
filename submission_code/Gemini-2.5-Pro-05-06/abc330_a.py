# YOUR CODE HERE
import sys

def main():
    # Read N and L from the first line
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    L = int(line1[1])

    # Read the list of scores A from the second line
    A_str = sys.stdin.readline().split()
    A = [int(s) for s in A_str]

    # Count how many scores are greater than or equal to L
    # Using a generator expression with sum() is efficient and Pythonic
    passed_count = sum(1 for score in A if score >= L)

    # Print the total number of students who passed
    print(passed_count)

if __name__ == '__main__':
    main()