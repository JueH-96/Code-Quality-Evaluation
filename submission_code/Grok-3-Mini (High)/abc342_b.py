import sys
data_iter = iter(sys.stdin.read().split())
N = int(next(data_iter))
position_map = [0] * (N + 1)
for pos in range(1, N + 1):
    person = int(next(data_iter))
    position_map[person] = pos
Q = int(next(data_iter))
for _ in range(Q):
    A = int(next(data_iter))
    B = int(next(data_iter))
    pos_A = position_map[A]
    pos_B = position_map[B]
    if pos_A < pos_B:
        print(A)
    else:
        print(B)