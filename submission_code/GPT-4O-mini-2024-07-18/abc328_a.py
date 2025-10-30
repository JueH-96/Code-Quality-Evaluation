# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    scores = list(map(int, data[2:2 + N]))
    
    total_score = sum(score for score in scores if score <= X)
    
    print(total_score)

if __name__ == "__main__":
    main()