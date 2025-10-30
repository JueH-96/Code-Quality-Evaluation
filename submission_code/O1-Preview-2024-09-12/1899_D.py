# YOUR CODE HERE
import sys
import threading
import math

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n = int(sys.stdin.readline())
        a = list(map(int, sys.stdin.readline().split()))
        freq = {}
        for val in a:
            v_i = math.log(val) - val * math.log(2)
            # Round to 8 decimal places
            v_i_rounded = round(v_i, 8)
            freq[v_i_rounded] = freq.get(v_i_rounded, 0) + 1
        total_pairs = 0
        for count in freq.values():
            total_pairs += count * (count -1) // 2
        print(total_pairs)
        
threading.Thread(target=main).start()