# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    takahashi_score = 0
    aoki_score = 0
    
    for i in range(1, N + 1):
        X, Y = map(int, data[i].split())
        takahashi_score += X
        aoki_score += Y
    
    if takahashi_score > aoki_score:
        print("Takahashi")
    elif aoki_score > takahashi_score:
        print("Aoki")
    else:
        print("Draw")

if __name__ == "__main__":
    main()