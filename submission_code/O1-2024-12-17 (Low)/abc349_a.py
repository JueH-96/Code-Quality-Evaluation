def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # The sum of all final scores must be 0 because each game adds +1 to the winner
    # and -1 to the loser. Therefore, if A_1 + ... + A_{N-1} = S, then A_N = -S.
    final_score_n = -sum(A)

    print(final_score_n)

# Do not forget to call main()
if __name__ == "__main__":
    main()