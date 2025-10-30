# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # The sum of all scores must be 0 because each game only transfers points between two players.
    # Therefore, the score of the Nth person can be calculated as the negative sum of the scores of the first N-1 persons.
    score_N = -sum(A)
    
    print(score_N)

if __name__ == "__main__":
    main()