# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    X = int(input[0])
    Y = int(input[1])
    Z = int(input[2])
    S = input[3]
    
    n = len(S)
    result = 0
    caps_lock_on = False
    
    i = 0
    while i < n:
        if S[i] == 'A':
            if caps_lock_on:
                result += X
            else:
                if i + 1 < n and S[i + 1] == 'A':
                    result += X
                else:
                    result += min(X + Z, Y)
                    caps_lock_on = not caps_lock_on
        else:  # S[i] == 'a'
            if not caps_lock_on:
                result += X
            else:
                if i + 1 < n and S[i + 1] == 'a':
                    result += X
                else:
                    result += min(X + Z, Y)
                    caps_lock_on = not caps_lock_on
        i += 1
    
    print(result)

if __name__ == "__main__":
    main()