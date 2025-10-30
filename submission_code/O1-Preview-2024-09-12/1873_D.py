# YOUR CODE HERE
import sys
import threading

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n, k = map(int, sys.stdin.readline().split())
        s = sys.stdin.readline().strip()
        positions = [i for i, c in enumerate(s) if c == 'B']
        ans = 0
        i = 0
        n_positions = len(positions)
        while i < n_positions:
            ans += 1
            last_covered = positions[i] + k -1
            i +=1
            while i < n_positions and positions[i] <= last_covered:
                i +=1
        print(ans)
    
threading.Thread(target=main).start()