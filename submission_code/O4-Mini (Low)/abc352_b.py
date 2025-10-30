import sys
import threading

def main():
    import sys
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    res = []
    i = 0  # pointer in S
    # scan T until we match all of S
    for j, ch in enumerate(T):
        if i < len(S) and ch == S[i]:
            # record 1-based position j+1
            res.append(j+1)
            i += 1
            if i == len(S):
                break
    
    # At this point, we should have matched all of S
    # Print the positions
    print(" ".join(map(str, res)))

if __name__ == "__main__":
    main()