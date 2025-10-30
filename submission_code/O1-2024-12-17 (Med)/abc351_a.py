def main():
    import sys
    data = list(map(int, sys.stdin.read().strip().split()))
    
    # The first 9 values are A_1..A_9 (Team Takahashi's runs each inning)
    A = data[:9]
    # The next 8 values are B_1..B_8 (Team Aoki's runs each inning except the 9th)
    B = data[9:]

    # Sum of Team Takahashi's runs through 9 innings
    score_takahashi = sum(A)
    # Sum of Team Aoki's runs through 8 innings
    score_aoki_8 = sum(B)

    # Team Aoki needs enough runs to surpass Team Takahashi's total score
    # So the minimum runs needed is: (score_takahashi - score_aoki_8) + 1
    runs_needed = (score_takahashi - score_aoki_8) + 1

    print(runs_needed)

# Do not forget to call main()!
if __name__ == "__main__":
    main()