# YOUR CODE HERE
def determine_winner(n):
    # If n is already divisible by 3, Vanya wins immediately
    if n % 3 == 0:
        return "First"
    
    # If n is 1 or 2 modulo 3, we need to check the number of moves
    moves = 0
    while n % 3 != 0 and moves < 10:
        n += 1 if n % 3 == 1 else -1
        moves += 1
        if n % 3 == 0:
            return "First"
        n += 1 if n % 3 == 1 else -1
        moves += 1
    
    # If 10 moves have passed and Vanya hasn't won, Vova wins
    return "Second"

t = int(input())
for _ in range(t):
    n = int(input())
    print(determine_winner(n))