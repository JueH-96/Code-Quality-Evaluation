import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    scores = data[1:]
    
    takahashi_total = 0
    aoki_total = 0
    
    for i in range(N):
        X = int(scores[2 * i])
        Y = int(scores[2 * i + 1])
        takahashi_total += X
        aoki_total += Y
    
    if takahashi_total > aoki_total:
        print("Takahashi")
    elif aoki_total > takahashi_total:
        print("Aoki")
    else:
        print("Draw")

if __name__ == "__main__":
    main()