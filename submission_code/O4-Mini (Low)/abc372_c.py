import sys
import threading

def main():
    import sys
    input = sys.stdin.readline
    
    N, Q = map(int, input().split())
    S = list(input().rstrip())
    
    # Helper to check if S[i:i+3] == "ABC"
    def is_abc(i):
        # i is 0-based start index
        return S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C'
    
    # Initial count
    cnt = 0
    for i in range(N-2):
        if is_abc(i):
            cnt += 1
    
    out = []
    for _ in range(Q):
        x, c = input().split()
        x = int(x) - 1  # convert to 0-based
        
        # For each possible affected substring start in [x-2, x]
        for start in (x-2, x-1, x):
            if 0 <= start <= N-3 and is_abc(start):
                cnt -= 1
        
        # Perform update
        S[x] = c
        
        # Recount affected substrings after update
        for start in (x-2, x-1, x):
            if 0 <= start <= N-3 and is_abc(start):
                cnt += 1
        
        out.append(str(cnt))
    
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()