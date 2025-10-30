import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:               # no input
        return
    N = int(data[0])           # number of people (not actually needed)
    scores = list(map(int, data[1:]))   # A_1 … A_{N-1}
    
    # The total score in the system is always zero because each game adds +1 and -1.
    # Therefore, score_N = - (score_1 + … + score_{N-1})
    answer = -sum(scores)
    print(answer)

if __name__ == "__main__":
    main()