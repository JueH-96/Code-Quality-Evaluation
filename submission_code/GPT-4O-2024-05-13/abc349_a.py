# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Calculate the sum of all given scores
    sum_A = sum(A)
    
    # Since the sum of all scores must be zero, the score of person N is -sum_A
    score_N = -sum_A
    
    print(score_N)

if __name__ == "__main__":
    main()