# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    A = int(input[0])
    M = int(input[1])
    L = int(input[2])
    R = int(input[3])

    # Find the first tree position >= L
    if L <= A:
        first_tree = A + M * ((L - A + M - 1) // M)
    else:
        first_tree = A + M * ((L - A) // M)

    # Find the last tree position <= R
    if R >= A:
        last_tree = A + M * ((R - A) // M)
    else:
        last_tree = A + M * ((R - A - M + 1) // M)

    # Calculate the number of trees between first_tree and last_tree inclusive
    if first_tree > R or last_tree < L:
        print(0)
    else:
        number_of_trees = (last_tree - first_tree) // M + 1
        print(number_of_trees)

if __name__ == "__main__":
    main()