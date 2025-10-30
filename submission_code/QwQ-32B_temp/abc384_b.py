import sys

def main():
    N, R = map(int, sys.stdin.readline().split())
    current_rating = R
    for _ in range(N):
        D, A = map(int, sys.stdin.readline().split())
        if D == 1:
            if 1600 <= current_rating <= 2799:
                current_rating += A
        else:
            if 1200 <= current_rating <= 2399:
                current_rating += A
    print(current_rating)

if __name__ == "__main__":
    main()