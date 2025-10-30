import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, T_prime = input().split()
    N = int(N)
    T_prime = T_prime.strip()
    m = len(T_prime)

    def is_within_one_edit(s: str, t: str) -> bool:
        # Check if edit distance between s and t is <= 1 under insert/delete/substitute
        n, m = len(s), len(t)
        # Quick length check
        if abs(n - m) > 1:
            return False
        # If lengths equal, allow at most one substitution
        if n == m:
            diff = 0
            for a, b in zip(s, t):
                if a != b:
                    diff += 1
                    if diff > 1:
                        return False
            return True
        # Ensure s is the shorter when handling insert-to-s case
        if n > m:
            # s longer by exactly 1 => t is shorter => can delete one char from s
            # Swap roles so that we skip in the longer string
            return is_within_one_edit(t, s)

        # Now n + 1 == m: t is longer by 1 => attempt to match by skipping one char in t
        i = j = 0
        skipped = False
        while i < n and j < m:
            if s[i] == t[j]:
                i += 1
                j += 1
            else:
                if skipped:
                    return False
                skipped = True
                # skip this char in the longer string t
                j += 1
        # Always allowable if we've reached the end with at most one skip
        return True

    result = []
    for idx in range(1, N + 1):
        s = input().strip()
        if is_within_one_edit(s, T_prime):
            result.append(idx)

    # Output
    print(len(result))
    if result:
        print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()