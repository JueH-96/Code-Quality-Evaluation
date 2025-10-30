def find_winner(n):
    if n % 3 == 0:
        return "First"
    else:
        return "Second"

t = int(input().strip())
for _ in range(t):
    n = int(input().strip())
    print(find_winner(n))