import math
import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print("No")
        return
        
    parts = data[0].split()
    N = int(parts[0])
    X = int(parts[1])
    Y = int(parts[2])
    S = data[1].strip()
    T = data[2].strip()
    
    ones_S = S.count('1')
    ones_T = T.count('1')
    if ones_S != ones_T:
        print("No")
        return
        
    d = math.gcd(X, Y)
    zeros_count_S = [0] * d
    zeros_count_T = [0] * d
    
    for i, char in enumerate(S):
        if char == '0':
            r = i % d
            zeros_count_S[r] += 1
            
    for i, char in enumerate(T):
        if char == '0':
            r = i % d
            zeros_count_T[r] += 1
            
    if zeros_count_S == zeros_count_T:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()