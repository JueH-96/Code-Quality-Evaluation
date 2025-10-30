def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    scores = list(map(int, data[1:]))

    # The sum of all final scores must be 0 (each game is zero-sum).
    # Therefore, the score of person N is the negative of the sum of the others.
    person_N_score = -sum(scores)

    print(person_N_score)

def main():
    solve()

if __name__ == "__main__":
    main()