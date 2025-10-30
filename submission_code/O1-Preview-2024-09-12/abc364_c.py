# YOUR CODE HERE
import sys
import threading

def main():
    import bisect

    N_str = sys.stdin.readline()
    while N_str.strip() == '':
        N_str = sys.stdin.readline()
    N, X_str, Y_str = N_str.strip().split()
    N = int(N)
    X = int(X_str)
    Y = int(Y_str)
    A_line = ''
    while len(A_line.strip().split()) < N:
        A_line += sys.stdin.readline()
    A_list = list(map(int, A_line.strip().split()))
    B_line = ''
    while len(B_line.strip().split()) < N:
        B_line += sys.stdin.readline()
    B_list = list(map(int, B_line.strip().split()))

    A = A_list
    B = B_list

    # Dishes sorted by decreasing sweetness
    dishes_sweet = sorted(zip(A, B), key=lambda x: -x[0])
    # Cumulative sums of sweetness
    cumsum_sweet = []
    total = 0
    for a, b in dishes_sweet:
        total += a
        cumsum_sweet.append(total)

    # Find minimal k_sweet such that cumsum_sweet[k-1] > X
    from bisect import bisect_right

    idx_sweet = bisect_right(cumsum_sweet, X)
    k_sweet = idx_sweet + 1  # Since k starts from 1
    if idx_sweet == len(cumsum_sweet):
        k_sweet = N + 1  # Sum never exceeds X

    # Dishes sorted by decreasing saltiness
    dishes_salt = sorted(zip(B, A), key=lambda x: -x[0])
    # Cumulative sums of saltiness
    cumsum_salt = []
    total = 0
    for b, a in dishes_salt:
        total += b
        cumsum_salt.append(total)

    idx_salt = bisect_right(cumsum_salt, Y)
    k_salt = idx_salt + 1
    if idx_salt == len(cumsum_salt):
        k_salt = N + 1

    min_dishes = min(k_sweet, k_salt)
    if min_dishes > N:
        min_dishes = N
    print(min_dishes)
    
threading.Thread(target=main).start()