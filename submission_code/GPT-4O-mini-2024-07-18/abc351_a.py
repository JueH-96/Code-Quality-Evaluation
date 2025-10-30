# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    A = list(map(int, data[:9]))
    B = list(map(int, data[9:]))
    
    # Calculate the total scores for both teams
    score_takahashi = sum(A)
    score_aoki = sum(B)
    
    # To win, Aoki needs to score more than Takahashi
    # Therefore, the minimum runs needed in the bottom of the ninth
    runs_needed = score_takahashi - score_aoki + 1
    
    print(runs_needed)

if __name__ == "__main__":
    main()