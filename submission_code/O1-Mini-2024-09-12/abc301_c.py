# YOUR CODE HERE
import sys

def main():
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    if len(S) != len(T):
        print("No")
        return
    
    allowed = set(['a','t','c','o','d','e','r'])
    
    counts_S = [0]*26
    counts_T = [0]*26
    s_at = 0
    t_at = 0
    
    for c in S:
        if c == '@':
            s_at +=1
        else:
            counts_S[ord(c)-97] +=1
    
    for c in T:
        if c == '@':
            t_at +=1
        else:
            counts_T[ord(c)-97] +=1
    
    # Check non-allowed characters
    for i in range(26):
        char = chr(i + 97)
        if char not in allowed:
            if counts_S[i] != counts_T[i]:
                print("No")
                return
    
    required_S = 0
    required_T = 0
    for c in allowed:
        idx = ord(c)-97
        if counts_S[idx] > counts_T[idx]:
            required_T += counts_S[idx] - counts_T[idx]
        else:
            required_S += counts_T[idx] - counts_S[idx]
    
    if required_S <= s_at and required_T <= t_at:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()