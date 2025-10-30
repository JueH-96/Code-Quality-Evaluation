# YOUR CODE HERE
import sys
from itertools import combinations

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    stands = input[2:]

    # Convert stands to a list of sets of flavors they sell
    stands = [set(i) for i in stands]

    # Try all combinations of stands and find the minimum number of stands that cover all flavors
    for k in range(1, N + 1):
        for combo in combinations(stands, k):
            if set.union(*combo) == set(range(M)):
                print(k)
                return

if __name__ == "__main__":
    main()