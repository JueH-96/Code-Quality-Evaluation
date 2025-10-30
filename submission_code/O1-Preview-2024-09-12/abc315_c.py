# YOUR CODE HERE
import sys
import threading

def main():
    import sys

    import math

    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    flavor_dict = {}
    cups = []
    for _ in range(N):
        F_i_str, S_i_str = sys.stdin.readline().split()
        F_i = int(F_i_str)
        S_i = int(S_i_str)
        # Build flavor to list of deliciousness mapping
        if F_i not in flavor_dict:
            flavor_dict[F_i] = []
        flavor_dict[F_i].append(S_i)
        cups.append((S_i, F_i))
    max_satisfaction = 0

    # For each flavor with at least two cups, compute s + t/2
    for S_i_list in flavor_dict.values():
        if len(S_i_list) >= 2:
            S_i_list.sort(reverse=True)
            s = S_i_list[0]
            t = S_i_list[1]
            satisfaction_same = s + t // 2
            if satisfaction_same > max_satisfaction:
                max_satisfaction = satisfaction_same

    # Now, find top K cups by deliciousness
    cups.sort(reverse=True)  # Sort by S_i in descending order
    K = 1000  # Adjust K as needed
    top_cups = cups[:K]
    n = len(top_cups)
    for i in range(n):
        s, f_s = top_cups[i]
        for j in range(i+1, n):
            t, f_t = top_cups[j]
            if f_s != f_t:
                satisfaction_diff = s + t
                if satisfaction_diff > max_satisfaction:
                    max_satisfaction = satisfaction_diff
    print(max_satisfaction)

if __name__ == '__main__':
    threading.Thread(target=main,).start()