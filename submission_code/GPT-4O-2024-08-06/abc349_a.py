# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Calculate the sum of A_1 to A_{N-1}
    sum_A = sum(A)
    
    # The score of person N is the negative of this sum
    score_N = -sum_A
    
    # Print the result
    print(score_N)

if __name__ == "__main__":
    main()