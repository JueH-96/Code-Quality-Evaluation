# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    X,Y,Z = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    N = len(S)
    INF = float('inf')

    dp = [[INF]*2 for _ in range(N+1)]
    # dp[pos][cap_state] = minimal cost to reach pos with cap_state (0: off, 1: on)
    dp[0][0] = 0  # Initially, Caps Lock is off
    dp[0][1] = INF

    for pos in range(N):
        for cap_state in [0,1]:
            if dp[pos][cap_state] == INF:
                continue

            # Option 1: Do not toggle Caps Lock
            cost_current = dp[pos][cap_state]

            # Cost to produce S[pos] with cap_state
            c = S[pos]
            if cap_state == 0:
                # Caps Lock is off
                if c == 'a':
                    # Need to produce 'a' with cap_state=0
                    # Press 'a' key (X ms)
                    cost_char = X
                elif c == 'A':
                    # Need to produce 'A' with cap_state=0
                    # Press 'a'+Shift (Y ms)
                    cost_char = Y
            else:
                # Caps Lock is on
                if c == 'a':
                    # Need to produce 'a' with cap_state=1
                    # Press 'a'+Shift (Y ms)
                    cost_char = Y
                elif c == 'A':
                    # Need to produce 'A' with cap_state=1
                    # Press 'a' key (X ms)
                    cost_char = X

            total_cost = cost_current + cost_char

            if dp[pos+1][cap_state] > total_cost:
                dp[pos+1][cap_state] = total_cost

            # Option 2: Toggle Caps Lock
            new_cap_state = 1 - cap_state
            cost_toggle = cost_current + Z  # Cost to toggle Caps Lock

            # Cost to produce S[pos] with new_cap_state
            if new_cap_state == 0:
                if c == 'a':
                    cost_char = X
                elif c == 'A':
                    cost_char = Y
            else:
                if c == 'a':
                    cost_char = Y
                elif c == 'A':
                    cost_char = X
            total_cost_toggle = cost_toggle + cost_char

            if dp[pos+1][new_cap_state] > total_cost_toggle:
                dp[pos+1][new_cap_state] = total_cost_toggle

    answer = min(dp[N][0], dp[N][1])
    print(answer)
threading.Thread(target=main).start()