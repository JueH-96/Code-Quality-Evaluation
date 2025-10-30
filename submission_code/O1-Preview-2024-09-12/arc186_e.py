# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N, M, K = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    mod = 998244353

    # Build failure function for KMP automaton
    fail = [0]*M
    for i in range(1, M):
        j = fail[i-1]
        while j > 0 and X[i] != X[j]:
            j = fail[j-1]
        if X[i] == X[j]:
            j += 1
        fail[i] = j

    # Build automaton transitions
    automaton = [{} for _ in range(M)]
    for state in range(M):
        for c in range(1, K+1):
            if c == X[state]:
                next_state = state +1
                if next_state == M:
                    # Reached the forbidden sequence, avoid this transition
                    next_state = -1
            else:
                next_state = state
                while next_state > 0 and c != X[next_state]:
                    next_state = fail[next_state-1]
                if c == X[next_state]:
                    next_state +=1
                # If next_state == M, avoid transition
                if next_state == M:
                    next_state = -1
            if next_state != -1:
                automaton[state][c] = next_state

    dp = [ [0]*M for _ in range(N+1) ]
    dp[0][0] =1

    for pos in range(N):
        for state in range(M):
            count = dp[pos][state]
            if count ==0:
                continue
            for c in range(1, K+1):
                if c in automaton[state]:
                    next_state = automaton[state][c]
                    dp[pos+1][next_state] = (dp[pos+1][next_state] + count)%mod

    ans = sum(dp[N])%mod
    print(ans)

threading.Thread(target=main).start()