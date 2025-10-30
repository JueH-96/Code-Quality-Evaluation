import sys

def solve():
    """
    Solves a single test case for the piece placement problem.
    """
    try:
        R, B = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        return

    # Condition checks based on parity analysis
    if R % 2 != 0:
        print("No")
        return
    if R == 0 and B % 2 != 0:
        print("No")
        return

    print("Yes")

    C = 500000  # A large constant offset for coordinates

    if R == 0:  # Case: R=0, B=2m (m>=1)
        m = B // 2
        # Construction of a diamond shape for blue pieces
        for i in range(m):
            print(f"B {C + i} {C + m - i}")
        for i in range(m):
            print(f"B {C + m - 1 - i} {C - 1 - i}")
    else:  # Case: R=2k (k>=1), B>=0
        # A robust construction for R>0 and even.
        # The pieces are ordered to form a valid cycle of moves.
        k = R // 2
        path = []

        # Start with two red pieces that will form the closing link of the cycle.
        # p_last(R) -> p_1(R)
        path.append(f"R {C} {C + 1}")  # p1
        path.append(f"R {C - 1} {C + 1}") # p_{R+B}

        # Chain of B blue pieces
        # p1 moves to p2 (first blue).
        # p1(R) at (C,C+1) can move to p2(B) at (C,C).
        for i in range(B):
            path.insert(1 + i, f"B {C + i} {C}")

        # Chain of remaining R-2 red pieces
        # The last blue piece p_{B+1} is at (C+B-1, C).
        # It moves to p_{B+2}, the first of the remaining reds.
        # p_{B+1}(B) -> p_{B+2}(R) must be a blue move.
        # So p_{B+2}(R) is at (C+B, C-1).
        for i in range(R - 2):
            path.insert(1 + B + 1 + i, f"R {C + B - i} {C - 1 - i}")

        for p in path:
            print(p)

def main():
    """
    Main function to handle multiple test cases.
    """
    try:
        num_test_cases_str = sys.stdin.readline()
        if not num_test_cases_str:
            return
        T = int(num_test_cases_str)
        for _ in range(T):
            solve()
    except (IOError, ValueError):
        # Gracefully handle empty input or malformed numbers
        pass

if __name__ == "__main__":
    main()