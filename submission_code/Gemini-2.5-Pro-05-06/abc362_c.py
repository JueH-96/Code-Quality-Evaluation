import sys

def solve():
    N = int(sys.stdin.readline())
    
    L_values = []
    R_values = []
    
    sum_L = 0
    sum_R = 0
    
    for _ in range(N):
        l, r = map(int, sys.stdin.readline().split())
        L_values.append(l)
        R_values.append(r)
        sum_L += l
        sum_R += r
        
    if sum_L <= 0 <= sum_R:
        print("Yes")
        
        # We start with X_i = L_i for all i. The sum is sum_L.
        # We need to increase this sum to 0.
        # The total amount to increase the sum by is (0 - sum_L) = -sum_L.
        # This amount must be non-negative because sum_L <= 0.
        amount_to_increase_sum = -sum_L
        
        # X_final will store the resulting sequence.
        X_final = [0] * N 
        
        for i in range(N):
            # For X_i, its initial value is L_values[i].
            # It can be increased up to R_values[i].
            # So, the maximum possible increment for X_i (from L_values[i])
            # is R_values[i] - L_values[i].
            max_increase_for_this_Xi = R_values[i] - L_values[i]
            
            # The actual increment for X_i is limited by two factors:
            # 1. How much we still need to increase the total sum (amount_to_increase_sum).
            # 2. How much this particular X_i can be increased (max_increase_for_this_Xi).
            # We take the minimum of these two.
            actual_increase_for_this_Xi = min(amount_to_increase_sum, max_increase_for_this_Xi)
            
            # The final value for X_i is its lower bound plus this increment.
            X_final[i] = L_values[i] + actual_increase_for_this_Xi
            
            # Reduce the remaining amount needed to increase the sum.
            amount_to_increase_sum -= actual_increase_for_this_Xi
            
        # After iterating through all X_i, amount_to_increase_sum should be 0
        # because the condition sum_L <= 0 <= sum_R guarantees that
        # -sum_L (total needed increase) <= sum_R - sum_L (total possible increase).
        # The greedy strategy fulfills the needed increase.
        
        # Print the resulting sequence X_final.
        # The `*` operator unpacks the list into individual arguments for print.
        print(*(X_final))
        
    else:
        # If sum_L > 0, the minimum possible sum is already > 0, so sum 0 is impossible.
        # If sum_R < 0, the maximum possible sum is already < 0, so sum 0 is impossible.
        print("No")

if __name__ == '__main__':
    solve()