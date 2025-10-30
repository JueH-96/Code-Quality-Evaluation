# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    moves = []
    pos_in_A = [0]*(N+1)  # positions of elements in A

    # Build position array, pos_in_A[val] = index of value val in A
    for idx, val in enumerate(A):
        pos_in_A[val] = idx

    for i in range(N):
        correct_value = i+1
        if A[i] != correct_value:
            idx_to_swap = pos_in_A[correct_value]
            # Swap A[i] and A[idx_to_swap]
            A[i], A[idx_to_swap] = A[idx_to_swap], A[i]
            moves.append((i+1, idx_to_swap+1))
            # Update positions
            pos_in_A[A[idx_to_swap]] = idx_to_swap
            pos_in_A[A[i]] = i

    print(len(moves))
    for move in moves:
        print(move[0], move[1])
threading.Thread(target=main).start()