# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N, W = map(int, sys.stdin.readline().split())
    items = []
    for _ in range(N):
        w_i, v_i = map(int, sys.stdin.readline().split())
        max_k_i = min((v_i + 1) // 2, W // w_i)
        for k_i in range(1, max_k_i + 1):
            delta_H = v_i - 2 * k_i + 1
            if delta_H <= 0:
                break
            items.append((delta_H, w_i))
    # Custom comparator to compare delta_H / w_i without floating point division
    def cmp(a):
        # Return (-delta_H / w_i, -delta_H)
        # Negative because we want to sort in decreasing order
        delta_H, w_i = a
        return (-delta_H * 1e9 // w_i, -delta_H)
    # Sort items in decreasing order of delta_H / w_i
    items.sort(key=lambda x: (-x[0] * 1e9 // x[1], -x[0]))
    total_weight = 0
    total_happiness = 0
    for delta_H, w_i in items:
        if total_weight + w_i <= W:
            total_weight += w_i
            total_happiness += delta_H
        else:
            continue
    print(total_happiness)
    

threading.Thread(target=main).start()