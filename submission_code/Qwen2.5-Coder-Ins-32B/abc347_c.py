# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = int(input[1])
    B = int(input[2])
    D = list(map(int, input[3:]))

    # The key observation is that if D_i % (A + B) <= A, then the i-th plan is on a holiday.
    # We need to check this for all plans.
    for d in D:
        if d % (A + B) > A:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()