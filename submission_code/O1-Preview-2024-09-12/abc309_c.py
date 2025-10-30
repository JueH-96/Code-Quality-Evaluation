# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import threading
    import bisect
    from collections import defaultdict

    sys.setrecursionlimit(1 << 25)

    N, K = map(int, sys.stdin.readline().split())
    total_pills = 0
    changes = defaultdict(int)
    a_max = 0
    for _ in range(N):
        a_i, b_i = map(int, sys.stdin.readline().split())
        total_pills += b_i
        changes[a_i + 1] -= b_i
        if a_i > a_max:
            a_max = a_i

    # Append day when all medications have ended
    D_inf = a_max + 1
    changes[D_inf] += 0  # Ensure D_inf is included in days

    # Initialize total_pills and sort the days
    days = sorted(changes.keys())

    if total_pills <= K:
        print(1)
        return

    for D in days:
        total_pills += changes[D]
        if total_pills <= K:
            print(D)
            return

    # After last day, total_pills remains same (should be zero)
    # So next day after max day
    print(D_inf + 1)

if __name__ == "__main__":
    threading.Thread(target=main).start()