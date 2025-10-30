import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    cakes_orig_data = []
    for i in range(N):
        x, y, z = map(int, sys.stdin.readline().split())
        # Store attributes and original index. Original index is useful for tie-breaking.
        cakes_orig_data.append(((x, y, z), i)) 

    # Define score functions (lambdas take a cake tuple: ((X,Y,Z), original_idx) )
    score_functions = []

    # Type 1: s_X*X + s_Y*Y + s_Z*Z for s_i in {-1, 1}
    for sx_sign in [-1, 1]:
        for sy_sign in [-1, 1]:
            for sz_sign in [-1, 1]:
                # Need to capture signs correctly in lambda using default arguments or a helper
                def make_score_func(sX, sY, sZ):
                    return lambda c_tuple: sX * c_tuple[0][0] + sY * c_tuple[0][1] + sZ * c_tuple[0][2]
                score_functions.append(make_score_func(sx_sign, sy_sign, sz_sign))

    # Type 2: Single attributes (X, Y, Z)
    score_functions.append(lambda c_tuple: c_tuple[0][0])  # X
    score_functions.append(lambda c_tuple: c_tuple[0][1])  # Y
    score_functions.append(lambda c_tuple: c_tuple[0][2])  # Z

    # Type 3: Negative single attributes (-X, -Y, -Z)
    score_functions.append(lambda c_tuple: -c_tuple[0][0]) # -X
    score_functions.append(lambda c_tuple: -c_tuple[0][1]) # -Y
    score_functions.append(lambda c_tuple: -c_tuple[0][2]) # -Z
    
    max_overall_total_price = 0

    for score_func in score_functions:
        # Create list of (score, original_idx, attributes) for sorting
        # attributes is (X,Y,Z) tuple
        # original_idx is used for tie-breaking to ensure stable sort behavior
        sortable_cakes = []
        for i in range(N):
            cake_tuple_with_idx = cakes_orig_data[i] # This is ((X,Y,Z), original_idx)
            current_score = score_func(cake_tuple_with_idx)
            sortable_cakes.append((current_score, cake_tuple_with_idx[1], cake_tuple_with_idx[0]))
        
        # Sort: primary key score (desc), secondary key original_idx (asc)
        sortable_cakes.sort(key=lambda item: (-item[0], item[1]))
        
        # Select top 2K cakes' attributes
        selected_cakes_attrs = [item[2] for item in sortable_cakes[:2*K]]
        
        current_pairing_total_price = 0
        for i in range(K):
            attrs_cake1 = selected_cakes_attrs[2*i]
            attrs_cake2 = selected_cakes_attrs[2*i+1]
            
            price = max(attrs_cake1[0] + attrs_cake2[0],
                        attrs_cake1[1] + attrs_cake2[1],
                        attrs_cake1[2] + attrs_cake2[2])
            current_pairing_total_price += price
        
        if current_pairing_total_price > max_overall_total_price:
            max_overall_total_price = current_pairing_total_price
            
    sys.stdout.write(str(max_overall_total_price) + "
")

num_test_cases = int(sys.stdin.readline())
for _ in range(num_test_cases):
    solve()