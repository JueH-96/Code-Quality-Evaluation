# YOUR CODE HERE
import sys
import math

def main():
    N_K = sys.stdin.readline().split()
    N = int(N_K[0])
    K = int(N_K[1])
    S = sys.stdin.readline().strip()

    total_strawberries = 0
    i = 0
    while i < N:
        if S[i] == 'O':
            # Start of a run
            run_start = i
            while i < N and S[i] == 'O':
                i +=1
            run_length = i - run_start
            strawberries_from_run = run_length // K
            total_strawberries += strawberries_from_run
        else:
            i +=1
    print(total_strawberries)

if __name__ == "__main__":
    main()