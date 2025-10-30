# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import bisect

    import math
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    T = []
    X = []
    for _ in range(N):
        t_i, x_i = map(int, sys.stdin.readline().split())
        T.append(t_i)
        X.append(x_i)
    N_PullTab = 0
    N_Regular = 0
    N_Opener = 0

    pull_tabs = []
    regular_cans = []
    can_openers = []

    for t_i, x_i in zip(T, X):
        if t_i == 0:
            pull_tabs.append(x_i)
        elif t_i ==1:
            regular_cans.append(x_i)
        else:
            can_openers.append(x_i)

    pull_tabs.sort(reverse=True)
    regular_cans.sort(reverse=True)
    can_openers.sort(reverse=True)

    N_PullTab = len(pull_tabs)
    N_Regular = len(regular_cans)
    N_Opener = len(can_openers)

    H_pt = [0]*(N_PullTab+1)
    for i in range(N_PullTab):
        H_pt[i+1] = H_pt[i] + pull_tabs[i]

    H_reg = [0]*(N_Regular+1)
    for i in range(N_Regular):
        H_reg[i+1] = H_reg[i] + regular_cans[i]

    C_cap = [0]*(N_Opener+1)
    for i in range(N_Opener):
        C_cap[i+1] = C_cap[i] + can_openers[i]

    max_happiness = 0
    M_limit = min(M, N_Opener)
    for O in range(0, M_limit+1):
        R = M - O
        K_cap = C_cap[O]
        K_max = min(K_cap, N_Regular, R)
        if K_max < 0:
            continue
        P_max = min(N_PullTab, R - K_max)

        total_happiness = H_reg[K_max] + H_pt[P_max]
        # Try to adjust K to maximize total happiness
        K = K_max
        P = R - K
        if P > N_PullTab:
            excess = P - N_PullTab
            K -= excess
            P = N_PullTab
            if K < 0:
                K = 0
                P = min(R, N_PullTab)

        if K > K_cap or K > N_Regular:
            K = min(K_cap, N_Regular)

        if K < 0:
            K = 0
        P = min(N_PullTab, R - K)

        total_happiness = H_reg[K] + H_pt[P]
        curr_K = K
        curr_P = P
        max_total_happiness = total_happiness

        # Since the marginal happiness is decreasing, try adjusting K to find better total happiness
        while K > 0:
            K -=1
            P +=1
            if P > N_PullTab or K < 0:
                break
            happiness = H_reg[K] + H_pt[P]
            if happiness > max_total_happiness:
                max_total_happiness = happiness
                curr_K = K
                curr_P = P
        if max_total_happiness > max_happiness:
            max_happiness = max_total_happiness

    # Also consider when O = min(N_Opener, M)
    O = min(N_Opener, M)
    R = M - O
    K_cap = C_cap[O]
    K_max = min(K_cap, N_Regular, R)
    if K_max >= 0:
        P_max = min(N_PullTab, R - K_max)
        total_happiness = H_reg[K_max] + H_pt[P_max]
        if total_happiness > max_happiness:
            max_happiness = total_happiness

    print(max_happiness)
        

threading.Thread(target=main).start()