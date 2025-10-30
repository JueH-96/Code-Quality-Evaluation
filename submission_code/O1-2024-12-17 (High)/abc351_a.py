def main():
    import sys

    # Read input values
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Calculate the total scores after the top of the 9th for Takahashi
    score_takahashi = sum(A)
    # Calculate the total score after the bottom of the 8th for Aoki
    score_aoki = sum(B)

    # To win, Aoki's final score must strictly exceed Takahashi's current score
    # Minimum runs needed = (score_takahashi - score_aoki) + 1
    runs_needed = (score_takahashi - score_aoki) + 1

    print(runs_needed)

# Do not forget to call main()        
if __name__ == "__main__":
    main()