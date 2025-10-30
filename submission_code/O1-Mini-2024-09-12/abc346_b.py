import sys
import math

def main():
    pattern = 'wbwbwwbwbwbw'
    W, B = map(int, sys.stdin.read().split())
    
    if W < 0 or B < 0 or (W + B) < 1 or W > 100 or B > 100:
        print("No")
        return
    
    total_length = W + B
    pattern_length = len(pattern)
    repeats = math.ceil(total_length / pattern_length) + 2
    S = pattern * repeats
    
    found = False
    for i in range(len(S) - total_length + 1):
        substring = S[i:i+total_length]
        count_w = substring.count('w')
        count_b = substring.count('b')
        if count_w == W and count_b == B:
            found = True
            break
    
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()