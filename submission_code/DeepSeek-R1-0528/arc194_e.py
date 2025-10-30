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
    
    zeros_S = S.count('0')
    zeros_T = T.count('0')
    if zeros_S != zeros_T:
        print("No")
        return
        
    g = math.gcd(X, Y)
    count_zeros_S = [0] * g
    count_zeros_T = [0] * g
    
    for i, char in enumerate(S):
        if char == '0':
            r = i % g
            count_zeros_S[r] += 1
            
    for i, char in enumerate(T):
        if char == '0':
            r = i % g
            count_zeros_T[r] += 1
            
    if count_zeros_S == count_zeros_T:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()