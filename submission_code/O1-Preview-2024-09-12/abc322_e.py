# YOUR CODE HERE
import sys
import math

def main():
    import sys
    import threading
    def solve():
        N, K, P = map(int, sys.stdin.readline().split())
        N = int(N)
        K = int(K)
        P = int(P)
        C = []
        A = []
        for _ in range(N):
            lst = list(map(int, sys.stdin.readline().split()))
            C_i = lst[0]
            A_i = lst[1:]
            C.append(C_i)
            A.append(A_i)
        P_plus_one = P + 1
        N_state = (P_plus_one) ** K
        INF = float('inf')
        dp = [INF] * N_state
        dp[0] = 0  # Initial state, all parameters are 0

        def get_index(state):
            index = 0
            for p in state:
                index = index * P_plus_one + p
            return index

        def get_state(index):
            state = [0] * K
            for i in reversed(range(K)):
                state[i] = index % P_plus_one
                index //= P_plus_one
            return state

        for state_index in range(N_state):
            current_cost = dp[state_index]
            if current_cost == INF:
                continue
            current_state = get_state(state_index)
            for i in range(N):
                plan_cost = C[i]
                plan_increase = A[i]
                new_state = [0]*K
                for j in range(K):
                    new_value = current_state[j] + plan_increase[j]
                    if new_value > P:
                        new_value = P
                    new_state[j] = new_value
                new_state_index = get_index(new_state)
                new_cost = current_cost + plan_cost
                if dp[new_state_index] > new_cost:
                    dp[new_state_index] = new_cost
        goal_state = [P]*K
        goal_index = get_index(goal_state)
        if dp[goal_index] == INF:
            print(-1)
        else:
            print(int(dp[goal_index]))
    threading.Thread(target=solve).start()