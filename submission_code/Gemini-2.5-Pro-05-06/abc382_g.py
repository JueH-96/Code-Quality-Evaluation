import math

def floor_div(a, b):
    # Ensures a/b is floored correctly for negative a
    return math.floor(a / b)

def get_tile_params_val(Px, Py, K):
    i = floor_div(Px, K)
    j = floor_div(Py, K)

    # is_H_type is True if i and j have the same parity
    is_H_type = ((i % 2) == (j % 2)) # More robust for negative i,j

    if is_H_type:
        k = Py - K * j
    else:
        k = Px - K * i
    return i, j, k, is_H_type

def calculate_k_cost(K, Is, Js, ks, is_Hs, It, Jt, kt, order_X_first):
    
    curr_k = ks
    curr_is_H = is_Hs
    k_cost = 0

    # Determine step directions
    # dx_step is 1 if It > Is, -1 if It < Is, 0 otherwise
    dx_step = 0
    if It > Is: dx_step = 1
    elif It < Is: dx_step = -1
    
    # dy_step is 1 if Jt > Js, -1 if Jt < Js, 0 otherwise
    dy_step = 0
    if Jt > Js: dy_step = 1
    elif Jt < Js: dy_step = -1

    if order_X_first:
        # X moves
        for _ in range(abs(Is - It)):
            if curr_is_H: # H type, move X direction -> Type A
                if dx_step == 1: # move right I -> I+1
                    curr_k = 0 
                else: # move left I -> I-1
                    curr_k = K - 1
            else: # V type, move X direction -> Type B
                if dx_step == 1: # move right I -> I+1, requires k_curr = K-1
                    k_cost += abs(curr_k - (K - 1))
                else: # move left I -> I-1, requires k_curr = 0
                    k_cost += abs(curr_k - 0)
                curr_k = 0 # Chosen k
            curr_is_H = not curr_is_H
        
        # Y moves
        for _ in range(abs(Js - Jt)):
            if curr_is_H: # H type, move Y direction -> Type B
                if dy_step == 1: # move up J -> J+1, requires k_curr = K-1
                    k_cost += abs(curr_k - (K-1))
                else: # move down J -> J-1, requires k_curr = 0
                    k_cost += abs(curr_k - 0)
                curr_k = 0 # Chosen k
            else: # V type, move Y direction -> Type A
                if dy_step == 1: # move up J -> J+1
                    curr_k = 0
                else: # move down J -> J-1
                    curr_k = K-1
            curr_is_H = not curr_is_H
    else: # Y moves first
        # Y moves
        for _ in range(abs(Js - Jt)):
            if curr_is_H: # H type, move Y direction -> Type B
                if dy_step == 1: 
                    k_cost += abs(curr_k - (K-1))
                else: 
                    k_cost += abs(curr_k - 0)
                curr_k = 0 
            else: # V type, move Y direction -> Type A
                if dy_step == 1: 
                    curr_k = 0
                else: 
                    curr_k = K-1
            curr_is_H = not curr_is_H
            
        # X moves
        for _ in range(abs(Is - It)):
            if curr_is_H: # H type, move X direction -> Type A
                if dx_step == 1: 
                    curr_k = 0
                else: 
                    curr_k = K - 1
            else: # V type, move X direction -> Type B
                if dx_step == 1: 
                    k_cost += abs(curr_k - (K-1))
                else: 
                    k_cost += abs(curr_k - 0)
                curr_k = 0 
            curr_is_H = not curr_is_H
            
    k_cost += abs(curr_k - kt)
    return k_cost

def solve():
    K, Sx, Sy, Tx, Ty = map(int, input().split())

    Is, Js, ks, is_Hs = get_tile_params_val(Sx, Sy, K)
    It, Jt, kt, is_Ht = get_tile_params_val(Tx, Ty, K)

    if Is == It and Js == Jt:
        print(abs(ks - kt))
        return

    # Calculate k-cost for path "X moves then Y moves"
    k_cost_1 = calculate_k_cost(K, Is, Js, ks, is_Hs, It, Jt, kt, True)
    
    # Calculate k-cost for path "Y moves then X moves"
    k_cost_2 = calculate_k_cost(K, Is, Js, ks, is_Hs, It, Jt, kt, False)
    
    min_k_cost = min(k_cost_1, k_cost_2)
    
    base_cell_moves = abs(Is - It) + abs(Js - Jt)
    print(base_cell_moves + min_k_cost)

num_test_cases = int(input())
for _ in range(num_test_cases):
    solve()