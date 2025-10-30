import sys
import threading

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    K = int(data[0])
    S = data[1]
    T = data[2]
    # Since K == 1, we just need to check if edit distance <= 1.
    if S == T:
        print("Yes")
        return
    n, m = len(S), len(T)
    # If lengths differ by more than 1, impossible with one edit.
    if abs(n - m) > 1:
        print("No")
        return
    # Case 1: same length -> check at most one replacement
    if n == m:
        diff = 0
        for a, b in zip(S, T):
            if a != b:
                diff += 1
                if diff > 1:
                    print("No")
                    return
        print("Yes")
        return
    # Case 2: T is one char longer -> check one insertion into S
    if n + 1 == m:
        i = j = 0
        diff = 0
        while i < n and j < m:
            if S[i] == T[j]:
                i += 1
                j += 1
            else:
                diff += 1
                if diff > 1:
                    print("No")
                    return
                j += 1
        # If we've passed the loop, it's okay even if the extra char is at the end.
        print("Yes")
        return
    # Case 3: S is one char longer -> check one deletion from S
    if m + 1 == n:
        i = j = 0
        diff = 0
        while i < n and j < m:
            if S[i] == T[j]:
                i += 1
                j += 1
            else:
                diff += 1
                if diff > 1:
                    print("No")
                    return
                i += 1
        print("Yes")
        return
    # All other cases:
    print("No")

if __name__ == "__main__":
    main()