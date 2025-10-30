import sys
from functools import lru_cache

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, *rest = map(int, sys.stdin.read().split())
    A = []
    B = []
    for i in range(N):
        A.append(rest[2*i])
        B.append(rest[2*i+1])

    @lru_cache(maxsize=None)
    def can_win(mask):
        # Find all available cards
        available = [i for i in range(N) if (mask & (1 << i))]
        # Try all possible pairs
        for i in range(len(available)):
            for j in range(i+1, len(available)):
                c1 = available[i]
                c2 = available[j]
                if A[c1] == A[c2] or B[c1] == B[c2]:
                    new_mask = mask & ~(1 << c1) & ~(1 << c2)
                    if not can_win(new_mask):
                        return True
        return False

    initial_mask = (1 << N) -1
    if can_win(initial_mask):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()