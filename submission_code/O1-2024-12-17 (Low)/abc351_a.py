def main():
    import sys
    data = list(map(int, sys.stdin.read().strip().split()))
    
    # First 9 integers are A_i
    A = data[:9]
    # Next 8 integers are B_j
    B = data[9:]
    
    scoreTakahashi = sum(A)
    scoreAoki = sum(B)
    
    # To win, Team Aoki needs (scoreTakahashi - scoreAoki) + 1 runs
    required_runs = (scoreTakahashi - scoreAoki) + 1

    print(required_runs)

# Do not forget to call main()
if __name__ == "__main__":
    main()