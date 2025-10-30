# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # The sum of all scores must be zero since each game's score change nets to zero.
    # Thus, sum(A) + score_N = 0
    # score_N = -sum(A)
    score_N = -sum(A)
    
    print(score_N)

if __name__ == "__main__":
    main()