def solve():
    import sys
    N_Q_and_instructions = sys.stdin.read().split()
    N = int(N_Q_and_instructions[0])
    Q = int(N_Q_and_instructions[1])
    instructions = N_Q_and_instructions[2:]
    left = 1
    right = 2
    total = 0

    def list_path_cw(N, C, T):
        steps_cw = (T - C + N) % N
        nodes_cw = []
        for s in range(1, steps_cw):
            node = (C + s -1) % N +1
            nodes_cw.append(node)
        return steps_cw, nodes_cw

    def list_path_ccw(N, C, T):
        steps_ccw = (C - T + N) % N
        nodes_ccw = []
        for s in range(1, steps_ccw):
            node = (C - s -1) % N +1
            nodes_ccw.append(node)
        return steps_ccw, nodes_ccw

    def min_moves(N, C, T, F):
        steps_cw, nodes_cw = list_path_cw(N, C, T)
        valid_cw = F not in nodes_cw
        steps_ccw, nodes_ccw = list_path_ccw(N, C, T)
        valid_ccw = F not in nodes_ccw
        inf = float('inf')
        distance_cw = steps_cw if valid_cw else inf
        distance_ccw = steps_ccw if valid_ccw else inf
        return min(distance_cw, distance_ccw)

    for i in range(Q):
        H_i = instructions[2*i]
        T_i = int(instructions[2*i +1])
        if H_i == 'L':
            C = left
            F = right
        else:
            C = right
            F = left
        # If already at target, no moves
        if C != T_i:
            d = min_moves(N, C, T_i, F)
            total +=d
            if H_i == 'L':
                left = T_i
            else:
                right = T_i
        # else, no move needed
    print(total)