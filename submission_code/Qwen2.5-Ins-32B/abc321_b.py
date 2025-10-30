import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:]))
    
    A.sort()
    min_score = X - sum(A[1:-1])
    
    if min_score > 100:
        print(-1)
    else:
        print(max(0, min_score))

if __name__ == "__main__":
    main()