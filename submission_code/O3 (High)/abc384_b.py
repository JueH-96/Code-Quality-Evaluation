import sys

def main() -> None:
    # Read N and initial rating R
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    rating = int(next(it))

    for _ in range(N):
        div = int(next(it))   # 1 or 2
        perf = int(next(it))  # A_i

        # Check whether Takahashi is subject to a rating update
        if div == 1:
            eligible = 1600 <= rating <= 2799
        else:  # div == 2
            eligible = 1200 <= rating <= 2399

        # Update rating if eligible
        if eligible:
            rating += perf

    # Output the final rating
    print(rating)

if __name__ == "__main__":
    main()