import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    R = int(data[ptr])
    ptr += 1
    contests = []
    for _ in range(N):
        D = int(data[ptr])
        ptr += 1
        A = int(data[ptr])
        ptr += 1
        contests.append((D, A))
    current_rating = R
    for D, A in contests:
        if D == 1:
            if 1600 <= current_rating <= 2799:
                current_rating += A
        elif D == 2:
            if 1200 <= current_rating <= 2399:
                current_rating += A
    print(current_rating)

if __name__ == "__main__":
    main()