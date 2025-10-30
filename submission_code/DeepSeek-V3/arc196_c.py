import sys

def solve():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N
    C = list(map(int, input[ptr:ptr+N]))
    
    # Collect indices where A[i] != B[i]
    flip_indices = []
    for i in range(N):
        if A[i] != B[i]:
            flip_indices.append(i)
    
    # Sort these indices based on C[i] in ascending order
    flip_indices.sort(key=lambda x: C[x])
    
    total_cost = 0
    current_sum = sum(A)
    
    for i in flip_indices:
        # The current A[i] is original A[i] before any flips. But during processing, each flip changes it.
        # Wait, no. The problem says that each operation is:
        # flip A_i, then pay sum A_k * C_k.
        # So the order is: flip, then compute cost based on new A.
        # So the sum is sum_{k} A_k after the flip.
        
        # So for each flip, the cost is (sum after flip) * C_i. But the flip affects the sum.
        # So the sum changes by (new A_i - old A_i) = (1 - old A_i) - old A_i = 1 - 2*old A_i.
        # So sum becomes current_sum + (1 - 2 * old A_i)
        
        # But the old A_i is the value before this flip. However, during processing, each flip is applied in sequence.
        # So for each flip in the list, the current A_i is the original A_i XOR number of times it's been flipped mod 2.
        # But since we process each flip once, the value is toggled once.
        
        # So for each i in flip_indices, the original A[i] is toggled once.
        original_val = A[i]
        new_val = 1 - original_val
        delta = new_val - original_val
        new_sum = current_sum + delta
        total_cost += new_sum * C[i]
        current_sum = new_sum
    
    print(total_cost)

solve()