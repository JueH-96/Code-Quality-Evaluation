def determine_winner(n):
    if n % 3 == 0:
        return "Second"
    return "First"

t = int(input())
for _ in range(t):
    n = int(input())
    print(determine_winner(n))