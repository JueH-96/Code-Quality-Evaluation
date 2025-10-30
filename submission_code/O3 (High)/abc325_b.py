import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    
    # total employees that can attend when the meeting starts at UTC hour t (0..23)
    attend = [0] * 24         

    for _ in range(N):
        w = int(next(it))     # number of employees
        x = int(next(it))     # local time at 0:00 UTC

        # local start time must be in [9, 17] for a one-hour meeting
        for local_start in range(9, 18):          # 9 .. 17 inclusive
            t = (local_start - x) % 24            # UTC hour that yields this local start
            attend[t] += w

    print(max(attend))

if __name__ == "__main__":
    main()