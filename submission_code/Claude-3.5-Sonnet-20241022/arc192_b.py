def solve():
    # Read input
    N = int(input())
    A = list(map(int, input().split()))
    
    # For each position, calculate how many moves can be made from it
    moves_per_pos = A.copy()
    
    # Calculate grundy numbers for each position
    # If moves_per_pos[i] is odd, grundy number is 1, else 0
    grundy = [moves % 2 for moves in moves_per_pos]
    
    # Calculate XOR sum of all grundy numbers
    xor_sum = 0
    for g in grundy:
        xor_sum ^= g
    
    # If XOR sum is 0, second player (Snuke) wins
    # If XOR sum is not 0, first player (Fennec) wins
    print("Snuke" if xor_sum == 0 else "Fennec")

solve()