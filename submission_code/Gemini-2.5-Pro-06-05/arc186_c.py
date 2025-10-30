import sys

def solve():
    """
    Solves a single test case for the Balls and Boxes game.
    """
    try:
        N, M = map(int, sys.stdin.readline().split())
        boxes_input = [sys.stdin.readline() for _ in range(N)]
    except (IOError, ValueError):
        return

    profitable_boxes = []
    for line in boxes_input:
        v, p = map(int, line.split())
        if v > p:
            profitable_boxes.append({'d': v - p, 'p': p})

    # Sort boxes by their potential profit (d) in descending order.
    # This is how Mr. Box would optimally choose boxes.
    profitable_boxes.sort(key=lambda x: x['d'], reverse=True)

    K = len(profitable_boxes)
    if K == 0:
        print(0)
        return

    # Precompute prefix sums for potential profits (S) and costs (C).
    S = [0] * (K + 1)
    C = [0] * (K + 1)
    for i in range(K):
        S[i + 1] = S[i] + profitable_boxes[i]['d']
        C[i + 1] = C[i] + profitable_boxes[i]['p']

    # Precompute stop_profit[k]: profit if the game ends right after k boxes are bought.
    # At that point, Mr. Ball would have given one ball for each type to force the purchase.
    stop_profit = [0] * (K + 1)
    for i in range(1, K + 1):
        stop_profit[i] = i - C[i]

    # W(k) represents the value of the subgame where k types are already established.
    # We solve this using backward induction (a form of DP).

    # Determine the maximum number of types that will be used in the game.
    M_limit = min(M, K)
    
    if M_limit == 0:
        print(0)
        return

    # Base case for the recurrence.
    # W is initialized with the value of the game at the maximum number of types.
    W = S[M_limit]
    
    # If M > K, Mr. Ball has an extra move after K boxes are bought.
    # He can introduce a (K+1)-th type, forcing Mr. Box to decline and end the game.
    # This changes the value of the game at state K.
    if M > K:
        # The game value at state K is min(S[K], stop_profit[K]).
        # The game is effectively limited to K types, so we adjust M_limit and W.
        W = min(S[K], stop_profit[K])
        M_limit = K
        if M_limit == 0:
            print(0)
            return

    # Apply the recurrence relation backwards from k = M_limit down to 2.
    # W(k-1) = min(S[k-1], max(stop_profit[k-1], W(k)))
    for k in range(M_limit, 1, -1):
        W = min(S[k - 1], max(stop_profit[k - 1], W))

    # The final answer is max(0, W) because Mr. Box can choose not to play at all
    # if the optimal outcome is a loss for him.
    print(max(0, W))


def main():
    """
    Main function to handle multiple test cases.
    """
    try:
        T_str = sys.stdin.readline()
        if not T_str: return
        T = int(T_str)
        for _ in range(T):
            solve()
    except (IOError, ValueError):
        return

if __name__ == "__main__":
    main()