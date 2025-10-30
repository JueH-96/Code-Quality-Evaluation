# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A = []
    B = []
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        A.append(a)
        B.append(b)
    dp = {}
    def can_win(state):
        if state in dp:
            return dp[state]
        # Check if there is any valid move
        cards = [i for i in range(N) if state & (1 << i)]
        for i in range(len(cards)):
            for j in range(i+1, len(cards)):
                idx_i = cards[i]
                idx_j = cards[j]
                if A[idx_i] == A[idx_j] or B[idx_i] == B[idx_j]:
                    next_state = state & ~(1 << idx_i) & ~(1 << idx_j)
                    if not can_win(next_state):
                        dp[state] = True
                        return True
        dp[state] = False
        return False
    initial_state = (1 << N) -1
    result = can_win(initial_state)
    if result:
        print('Takahashi')
    else:
        print('Aoki')
threading.Thread(target=main).start()